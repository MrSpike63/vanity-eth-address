# Vanity Eth Address
Vanity Eth Address is a tool to generate Ethereum addresses that match certain criteria, accelerated with NVIDIA CUDA-enabled GPUs.

## Usage
```
./vanity-eth-addresss [PARAMETERS]
    Scoring methods
      (-lz) --leading-zeros               Count zero bytes at the start of the address
       (-z) --zeros                       Count zero bytes anywhere in the address
    Modes (normal addresses by default)
       (-c) --contract                    Search for addresses and score the contract address generated using nonce=0
      (-c2) --contract2                   Search for contract addresses using the CREATE2 opcode
      (-c3) --contract3                   Search for contract addresses using a CREATE3 proxy deployer
    Other:
       (-d) --device <device_number>      Use device <device_number> (Add one for each device for multi-gpu)
       (-b) --bytecode <filename>         File containing contract bytecode (only needed when using --contract2 or --contract3)
       (-a) --address <address>           Sender contract address (only needed when using --contract2 or --contract3)
      (-ad) --deployer-address <address>  Deployer contract address (only needed when using --contract3)
       (-w) --work-scale <num>            Defaults to 15. Scales the work done in each kernel. If your GPU finishes kernels within a few seconds, you may benefit from increasing this number.

Examples:
    ./vanity-eth-address --zeros --device 0 --device 2 --work-scale 17
    ./vanity-eth-address --leading-zeros --contract2 --bytecode bytecode.txt --address 0x0000000000000000000000000000000000000000 --device 0
```

## Benchmarks
| GPU  | Normal addresses | Contract addresses | CREATE2 addresses |
| ---- | ---------------- | ------------------ | ----------------- |
| 4090 | 3800M/s          | 2050M/s            | 4800M/s           |
| 3090 | 1600M/s          | 850M/s             | 2300M/s           |
| 3070 | 1000M/s          | 550M/s             | 1300M/s           |

Note that configuration and environment can affect performance.

## Requirements
* A NVIDIA CUDA-enabled GPU with a compute capability of at least 5.2 (Roughly anything above a GeForce GTX 950. For a full list [see here](https://developer.nvidia.com/cuda-gpus)).