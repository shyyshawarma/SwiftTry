import torch
import time

# Set parameters
batch_size, num_tokens, c = 32, 4096, 256  # Large tensor for a more realistic benchmark
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Initialize tensors
source_tensor = torch.randn(batch_size * num_tokens, c, device=device).half()
target_tensor = torch.zeros(batch_size * num_tokens, c, device=device).half()

# Random indices for testing
indices = torch.randperm(batch_size * num_tokens)[:batch_size * num_tokens // 2].to(device)

### Direct Indexing Benchmark
start_time = time.time()
for _ in range(1000):

    target_tensor[indices] = source_tensor[indices]
    torch.cuda.synchronize() if device == "cuda" else None  # Ensure GPU operations complete
direct_indexing_time = time.time() - start_time

### torch.index_select Benchmark
start_time = time.time()
for _ in range(1000):
    selected_tensor = torch.index_select(source_tensor, 0, indices)
    target_tensor.index_copy_(0, indices, selected_tensor)
    torch.cuda.synchronize() if device == "cuda" else None  # Ensure GPU operations complete
index_select_time = time.time() - start_time

# Display results
print(f"Direct indexing time: {direct_indexing_time:.6f} seconds")
print(f"torch.index_select time: {index_select_time:.6f} seconds")


print("Direct/index_select", direct_indexing_time/index_select_time)