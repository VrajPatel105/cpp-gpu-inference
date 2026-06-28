// This problem is given by claude. I am going to rederive this whole kernel without any reference of past code.


// //Cold re-derivation roadmap (blank file, no notes)

// The seed. What is one thread responsible for producing? (Everything below depends on this one sentence — write it first.)
// Identity. What two CUDA built-ins identify a thread, and what's the rule for which family (.x / .y) maps to row vs col? State the rule before you use it.
// Ownership. From the badge, which P element is mine? Write row and col — each is badge × TILE_WIDTH + offset. Get the pairing right without looking.
// Storage. Three declarations. Which get __shared__ and why? Which one is a private register and why? Tie each to its scope.
// Phases. Write the loop bound. Then say out loud: a phase is a chunk of ____ (time / space?), and what does one phase accomplish for a single thread?
// Load. Write both load indices flattened. The test: which one carries the per-phase jump on the column, which on the row, and why? (M slides →, N slides ↓.)
// Barrier #1. Why here? What bug does it stop? (Your mailbox / box3 story.)
// Inner k-loop. Write the accumulate line. Why is it a loop and not one statement?
// Barrier #2, then write home. Why a second barrier? What's the final write, flattened?

// Verification: trace thread (row=2, col=1), hand-compute its element, run, check h_C[9].
// Twists (after the clean version works)

// Easy — kill the symmetry. Right now A == B, so a row/col swap bug could hide. Make B different from A and re-verify by hand. If your output still matches, your row/col indexing is genuinely correct, not accidentally.
// Medium — break the square. The book (and your kernel) assumes N×N. Real matmul is A[I×K] · B[K×J] = P[I×J]. You've been using one N for three different dimensions. Which of your indices actually need I, K, J instead? This forces you to see the three dimensions you've been collapsing into one.
// Hard — break the seed. What if each thread computed two P elements instead of one? (This is real — it's called thread coarsening / register tiling, a genuine optimization PMPP hits later.) What changes in the launch, the loop, the accumulators? It deliberately violates "one thread, one element," which is exactly why wrestling with it deepens the rule.