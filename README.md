# devwrap

A design pattern for matching the device (cpu or gpu) of a torch function's inputs
to tensors creating inside the function. If code creates lots of tensors inside
a function, it is tedious to manually match the device of the tensors to the device
of the inputs. This pattern abstracts the problem to an analysis of the function's
inputs.

The pattern can be extended to tensor `dtype` and `layout` as well, but these
may require more care to make work in all cases.

The `test_wrapper.py` file contains two tests demonstrating the utility of the
pattern. It shows also that docstrings of the original function are preserved
in the wrapped function.

To run the example, clone the repo and run `pytest` in the root directory. It
requires `pytest` to run the tests and `torch` with a CUDA build. If these are
all satisfied, the repo code can be run as

```sh
pip install -e .
pytest
```
