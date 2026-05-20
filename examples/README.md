
NOTE: These examples are used to show how to call c API from other cpp files,
      and it does not include any functional checking.

ENVs:
```
export MACA_PATH=/your/maca/path

```

Make and run:
`````
make

#export LD_LIBRARY_PATH=/path/to/your/libmcFlashAttn.so/path:$MACA_PATH/lib:$MACA_PATH/ompi/lib:$LD_LIBRARY_PATH
./build/flash_attn_fwd_example
./build/flash_attn_varlen_fwd_example
./build/flash_attn_fwd_kvcache_example
./build/flash_attn_bwd_example
```
