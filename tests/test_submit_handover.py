"""The submit handover (gpu/vk/vk_replay.cpp): a recording's submit that comes while a present has the queue is left with
the presenting thread, which sends it once its present is back.

handover_begin, handover_leave, handover_end and handover_sent are taken from the source as they are and run against a queue
of the test's own (submit_now notes what is sent, by which thread, with the queue's lock held or not):

- with no present under way nothing is left: the submit is its caller's
- what is left during a present is sent by the presenting thread, in the order it came, before it lets the queue go; the
  next thread to take the queue's lock finds nothing left
- every submit reaches the queue once, in the order the submitting thread made them, whether left or sent directly, over
  20,000 submits racing 4,000 presents and a third thread that takes the queue between them
- a submit left with a present is not waited for before it is sent (handover_sent holds until the present lets go), and one
  that failed there says so

python -I -B tests/test_submit_handover.py [--cxx g++]
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def block(text, marker):
    start = text.index(marker)
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--source', type=Path, default=ROOT / 'gpu/vk/vk_replay.cpp')
    args = parser.parse_args()
    text = args.source.read_text(encoding='utf-8')
    handed = block(text, 'struct Handed {') + ';'
    helper = chr(10).join(block(text, marker) for marker in (
        'void handover_begin() {',
        'bool handover_leave(const Handed& h) {',
        'void handover_end() {',
        'bool handover_sent(Recording& r) {'))
    fixture = r'''
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#define CHECK(ok) do { if (!(ok)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #ok); std::exit(1); } } while (0)
// (what the functions name of Vulkan and of the replay, as the test's own)
struct VkCommandBufferSubmitInfo { int sType; void* commandBuffer; };
struct VkSemaphoreSubmitInfo { int sType; uint64_t value; };
struct VkSubmitInfo2 {   // (the rest zero when only sType is given, as the source writes it)
    int sType;
    uint32_t commandBufferInfoCount = 0;
    const VkCommandBufferSubmitInfo* pCommandBufferInfos = nullptr;
    uint32_t signalSemaphoreInfoCount = 0;
    const VkSemaphoreSubmitInfo* pSignalSemaphoreInfos = nullptr;
};
constexpr int VK_STRUCTURE_TYPE_SUBMIT_INFO_2 = 7;
struct Recording {
    uint64_t value = 0;
    bool submitted = false;
    std::atomic<uint8_t> handed{0};
};
struct { std::mutex present_mu; } dev;
void logf(const char*, ...) {}
bool handover_on() { return true; }
''' + handed + r'''
std::mutex handed_mu;
std::vector<Handed> handed;
bool presenting = false;
std::atomic<int64_t> st_handed{0}, st_handable{0};
// (the queue: what was sent, in order; who has its lock; a value whose submit fails)
std::vector<uint64_t> sent;
std::atomic<std::thread::id> queue_owner{};
uint64_t fails = 0;
int64_t sent_handed = 0;
void submit_now(const VkSubmitInfo2& si, Recording& r, bool was_handed) {
    CHECK(queue_owner.load() == std::this_thread::get_id());   // (the queue is the calling thread's)
    CHECK(si.sType == VK_STRUCTURE_TYPE_SUBMIT_INFO_2 && si.commandBufferInfoCount == 1 && si.pCommandBufferInfos[0].commandBuffer == &r);
    CHECK(si.signalSemaphoreInfoCount == 1 && si.pSignalSemaphoreInfos[0].value == r.value);   // (its own signal with it)
    sent.push_back(r.value);
    sent_handed += was_handed;
    const bool ok = r.value != fails;
    if (was_handed) r.handed.store(ok ? 0 : 2);
    else r.submitted = ok;
}
''' + helper + r'''
struct QueueLock {   // (dev.present_mu, with its owner noted)
    std::lock_guard<std::mutex> lk{dev.present_mu};
    QueueLock() { queue_owner.store(std::this_thread::get_id()); }
    ~QueueLock() { queue_owner.store(std::thread::id{}); }
};
Handed make(Recording& r) {
    Handed h{};
    h.cb[0] = {0, &r};
    h.sig[0] = {0, r.value};
    h.n = 1, h.nsig = 1, h.r = &r;
    return h;
}
// A recording's submit as submit_recording makes it: left with a present, or sent under the queue's lock
void submit(Recording& r) {
    const Handed h = make(r);
    if (handover_leave(h)) return;
    QueueLock q;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, h.n, h.cb, h.nsig, h.sig};
    submit_now(si, r, false);
}
bool nothing_left() {
    std::lock_guard hk(handed_mu);
    return handed.empty() && !presenting;
}
int main() {
    {   // no present under way: nothing is left, the submit is the caller's
        Recording r;
        r.value = 1;
        submit(r);
        CHECK(sent == std::vector<uint64_t>{1} && r.submitted && r.handed == 0 && sent_handed == 0 && nothing_left());
    }
    {   // during a present: left, in order, sent by the presenting thread before it lets the queue go
        sent.clear();
        std::vector<Recording> rs(4);
        for (uint64_t i = 0; i < 4; i++) rs[i].value = 10 + i;
        fails = 12;
        {
            QueueLock q;
            handover_begin();
            for (int i = 0; i < 3; i++) submit(rs[i]);
            CHECK(sent.empty() && rs[0].submitted && rs[0].handed == 1 && rs[2].handed == 1);
            handover_end();
            CHECK(sent == (std::vector<uint64_t>{10, 11, 12}) && sent_handed == 3 && nothing_left());
        }
        submit(rs[3]);   // (after it: its caller's again)
        CHECK(sent.back() == 13 && sent_handed == 3);
        CHECK(handover_sent(rs[0]) && handover_sent(rs[1]) && !handover_sent(rs[2]) && handover_sent(rs[3]));   // (12 failed there)
        CHECK(rs[2].handed == 0);
        fails = 0;
    }
    {   // one left with a present is not waited for before that present has sent it
        sent.clear(), sent_handed = 0;
        Recording r;
        r.value = 20;
        std::atomic<int> stage{0};
        std::thread waiter;
        {
            QueueLock q;
            handover_begin();
            submit(r);
            waiter = std::thread([&] {
                stage = 1;
                CHECK(handover_sent(r));   // (holds until the present lets the queue go)
                CHECK(stage.load() == 2 && sent == std::vector<uint64_t>{20});
                stage = 3;
            });
            while (stage.load() != 1) std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            CHECK(stage.load() == 1);   // (still waiting: the present has the queue)
            stage = 2;
            handover_end();
        }
        waiter.join();
        CHECK(stage.load() == 3);
    }
    {   // submits racing presents, and a third thread that takes the queue between them
        sent.clear(), sent_handed = 0;
        constexpr uint64_t kSubmits = 20000;
        constexpr int kPresents = 4000;
        std::vector<Recording> rs(kSubmits);
        std::atomic<bool> done{false};
        std::atomic<uint64_t> presents{0};
        std::thread presenter([&] {
            for (int i = 0; i < kPresents || !done.load(); i++) {
                {
                    QueueLock q;
                    handover_begin();
                    for (int spin = 0; spin < (i % 7) * 40; spin++) std::this_thread::yield();   // (the present, of several lengths)
                    handover_end();
                    CHECK(nothing_left());   // (before the queue is let go)
                }
                presents++;
                if (i % 3 == 0) std::this_thread::yield();
            }
        });
        std::thread other([&] {   // (a readback's submit, a sparse bind, a device wait: the queue's lock, taken blocking)
            while (!done.load()) {
                {
                    QueueLock q;
                    CHECK(nothing_left());   // (whoever takes the queue after a present finds the list sent)
                }
                std::this_thread::yield();
            }
        });
        for (uint64_t i = 0; i < kSubmits; i++) {
            rs[i].value = 100 + i;
            submit(rs[i]);
            if (i >= 48) CHECK(handover_sent(rs[i - 48]));   // (the slot that comes round, as recording_wait asks)
            while (presents.load() * 5 < i) std::this_thread::yield();   // (five submits a present: neither runs away from the other)
        }
        done = true;
        presenter.join(), other.join();
        CHECK(nothing_left() && sent.size() == kSubmits);
        for (uint64_t i = 0; i < kSubmits; i++) CHECK(sent[i] == 100 + i);   // (each once, in the order they were made)
        for (uint64_t i = 0; i < kSubmits; i++) CHECK(rs[i].submitted && rs[i].handed.load() == 0);
        std::printf("%lld of %llu submits were left with a present\n", (long long)sent_handed, (unsigned long long)kSubmits);
    }
    std::puts("PASS: submits left with a present are sent once, in order, before the queue is let go");
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'handover.cpp'
        exe = Path(tmp) / 'handover.exe'
        src.write_text(fixture, encoding='utf-8')
        subprocess.run([args.cxx, '-std=c++20', '-O1', '-Wall', '-Wextra', '-Werror', '-pthread', str(src), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True, timeout=120)


if __name__ == '__main__':
    main()
