# PIConGPU on Perlmutter
## General Remarks
PIConGPU can be compiled on a login node, but remember to limit the number of used CPU cores. `pic-build -j 16` works fine; reduce the number if the compiler gets killed.

## Installing Dependencies
Missing dependencies can be installed from source with `dependencies_autoinstall.sh`. Change the project number in the `gpu.profile`, source it, and execute the install script on a login node.

## Preemptive Jobs
When using preemptive queues, you should uncomment the `sleep 120` command at the end of the tpl file.

## Streaming
Note: This is quite an old example; some things may have changed. `gpu_stream.tpl` is an example streaming configuration. It needs to be adjusted (CPU distribution between PIConGPU and reader) to individual needs. The reader has to be provided as `input/bin/reader`. The minimal writer (PIConGPU)/reader JSON OpenPMD configuration is:
```json
{
  "adios2": {
    "engine": {
      "parameters": {
        "DataTransport": "rdma"
      }
    }
  }
}
```
Currently, this requires at least two nodes to ensure that the network interface is available.
