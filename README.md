# `NIfTI` file reader and writer

Lightweight `numpy` extension wrapper around [`nifti_clib`](https://github.com/NIFTI-Imaging/nifti_clib).

Blazingly fast &#x1F680;.

```Python
import nifti

nifti.read_volume('file.nii.gz')
nifti.read_header('file.nii.gz')
```

## Benchmarks

### Read header

| scenario        | exec         |     time |   relative |    it |     it/s |
|:----------------|:-------------|---------:|-----------:|------:|---------:|
| 3D uncompressed | nibabel      | 3.50411  |    1       | 10000 |  2853.79 |
| 3D uncompressed | nifti (this) | 0.242236 |   14.4657  | 10000 | 41282.1  |
| 3D compressed   | nibabel      | 4.35458  |    1       | 10000 |  2296.43 |
| 3D compressed   | nifti (this) | 0.470563 |    9.25397 | 10000 | 21251.1  |

### Read volume

| scenario        | exec         |     time |   relative |   it |    it/s |
|:----------------|:-------------|---------:|-----------:|-----:|--------:|
| 3D uncompressed | nibabel      | 0.376704 |    1       |   20 | 53.0921 |
| 3D uncompressed | nifti (this) | 0.254842 |    1.47819 |   20 | 78.4801 |
| 3D compressed   | nibabel      | 4.70112  |    1       |   20 |  4.2543 |
| 3D compressed   | nifti (this) | 0.265065 |   17.7357  |   20 | 75.4531 |
