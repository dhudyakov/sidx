KeyIndex 

A high-performance lookup utility designed to scan contiguous key storage using AVX-512.

Features

Parallel SIMD Comparison: Supports direct lookups of keys that are 1, 2, 4, or 8 bytes in size. AVX-512 evaluates all key lanes concurrently.

Fast Lookups: For data blocks fitting within one AVX-512 register, the lookup resolves in about 3 CPU instructions per block, exactly 1 vector load, 1 parallel comparison, and 1 bitmask scan instruction, eliminating pointer chasing and branching. 3 execution units can scan 3 blocks at a time.

Compact & Dense Memory: Keys are packed into a dense, contiguous memory block with zero node pointers, zero bucket chain pointers, and zero internal padding between elements.

Split Map Implementation: The slot index yielded by the search acts as a coordinate that can be paired directly with a separate, parallel payload buffer to implement a split map.

