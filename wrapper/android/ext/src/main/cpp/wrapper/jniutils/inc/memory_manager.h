#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <new>
#include <cassert>

class MemoryManager {
public:
    struct PoolHeader {
        MemoryManager *pool;
    };

    MemoryManager(std::size_t blockSize, std::size_t blockCount);

    template<typename T, typename... Args>
    std::unique_ptr<T, void (*)(T*)> create(Args&&... args);

private:
    std::vector<unsigned char> data_;
    unsigned char *freeListHead_{nullptr};

    std::size_t userBlockSize_;
    std::size_t totalBlockSize_;
    std::size_t blockCount_;

    template<typename T>
    static void deleter(T *obj);
    void *allocateBlock();
    void deallocateBlock(void *ptr);
};

