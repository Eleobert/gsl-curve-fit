# C++ Wrapper of gsl non linear least squares

Easy curve fit using gsl non linear least squares:

```C++
auto result = curve_fit(gaussian, {1.0, 0.0, 1.0}, xs, ys);
```

The example in the `example.cpp` file is adapted from the [gsl webpage](https://www.gnu.org/software/gsl/doc/html/nls.html#geodesic-acceleration-example-2).