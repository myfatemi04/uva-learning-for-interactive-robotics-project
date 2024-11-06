import os
# for i in range(20):
#     os.system(f"wget -P data/mt80 https://huggingface.co/datasets/nicklashansen/tdmpc2/resolve/main/mt80/chunk_{i}.pt")

for i in range(4):
    os.system(f"wget -P data/mt30 https://huggingface.co/datasets/nicklashansen/tdmpc2/resolve/main/mt30/chunk_{i}.pt")