#include <stdio.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "nifti1_io.h"

static PyObject *niftilib_read_header_c(const PyObject *self, PyObject *args)
{
    char *param_filename = NULL;

    if (!PyArg_ParseTuple(args, "s", &param_filename))
        return NULL;

    nifti_1_header *nih = nifti_read_header(param_filename, NULL, 0);
    if (nih == NULL)
    {
        return NULL;
    }

    PyObject *re = Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
        "sizeof_hdr", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, &nih->sizeof_hdr, 0, NPY_ARRAY_CARRAY, NULL),           /* MUST be 348 */
        "data_type", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, &nih->data_type, 4, NPY_ARRAY_CARRAY, NULL),            /* ++UNUSED++ */
        "db_name", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, &nih->db_name, 4, NPY_ARRAY_CARRAY, NULL),                /* ++UNUSED++ */
        "extents", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, &nih->extents, 0, NPY_ARRAY_CARRAY, NULL),                 /* ++UNUSED++ */
        "session_error", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, &nih->session_error, 0, NPY_ARRAY_CARRAY, NULL),     /* ++UNUSED++ */
        "regular", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, &nih->regular, 0, NPY_ARRAY_CARRAY, NULL),                  /* ++UNUSED++ */
        "dim_info", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, &nih->dim_info, 0, NPY_ARRAY_CARRAY, NULL),                /* MRI slice ordering. */
        "dim", PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_INT16, NULL, &nih->dim, 0, NPY_ARRAY_CARRAY, NULL),                         /* Data array dimensions. */
        "intent_p1", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->intent_p1, 0, NPY_ARRAY_CARRAY, NULL),           /* 1st intent parameter. */
        "intent_p2", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->intent_p2, 0, NPY_ARRAY_CARRAY, NULL),           /* 2nd intent parameter. */
        "intent_p3", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->intent_p3, 0, NPY_ARRAY_CARRAY, NULL),           /* 3rd intent parameter. */
        "intent_code", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, &nih->intent_code, 0, NPY_ARRAY_CARRAY, NULL),         /* NIFTI_INTENT_* code. */
        "datatype", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, &nih->datatype, 0, NPY_ARRAY_CARRAY, NULL),               /* Defines data type! */
        "bitpix", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, &nih->bitpix, 0, NPY_ARRAY_CARRAY, NULL),                   /* Number bits/voxel */
        "slice_start", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, &nih->slice_start, 0, NPY_ARRAY_CARRAY, NULL),         /* First slice index. */
        "pixdim", PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_FLOAT32, NULL, &nih->pixdim, 0, NPY_ARRAY_CARRAY, NULL),                 /* Grid spacings. */
        "vox_offset", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->vox_offset, 0, NPY_ARRAY_CARRAY, NULL),         /* Offset into .nii file */
        "scl_slope", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->scl_slope, 0, NPY_ARRAY_CARRAY, NULL),           /* Data scaling: slope. */
        "scl_inter", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->scl_inter, 0, NPY_ARRAY_CARRAY, NULL),           /* Data scaling: offset. */
        "slice_end", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, &nih->slice_end, 0, NPY_ARRAY_CARRAY, NULL),             /* Last slice index. */
        "slice_code", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, &nih->slice_code, 0, NPY_ARRAY_CARRAY, NULL),            /* Slice timing order. */
        "xyzt_units", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, &nih->xyzt_units, 0, NPY_ARRAY_CARRAY, NULL),            /* Units of pixdim[1..4] */
        "cal_max", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->cal_max, 0, NPY_ARRAY_CARRAY, NULL),               /* Max display intensity */
        "cal_min", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->cal_min, 0, NPY_ARRAY_CARRAY, NULL),               /* Min display intensity */
        "slice_duration", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->slice_duration, 0, NPY_ARRAY_CARRAY, NULL), /* Time for 1 slice. */
        "toffset", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->toffset, 0, NPY_ARRAY_CARRAY, NULL),               /* Time axis shift. */
        "glmax", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, &nih->glmax, 0, NPY_ARRAY_CARRAY, NULL),                     /* ++UNUSED++ */
        "glmin", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, &nih->glmin, 0, NPY_ARRAY_CARRAY, NULL),                     /* ++UNUSED++ */
        "descrip", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, &nih->descrip, 4, NPY_ARRAY_CARRAY, NULL),                /* any text you like. */
        "aux_file", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, &nih->aux_file, 4, NPY_ARRAY_CARRAY, NULL),              /* auxiliary filename. */
        "qform_code", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, &nih->qform_code, 0, NPY_ARRAY_CARRAY, NULL),           /* NIFTI_XFORM_* code. */
        "sform_code", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, &nih->sform_code, 0, NPY_ARRAY_CARRAY, NULL),           /* NIFTI_XFORM_* code. */
        "quatern_b", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->quatern_b, 0, NPY_ARRAY_CARRAY, NULL),           /* Quaternion b param. */
        "quatern_c", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->quatern_c, 0, NPY_ARRAY_CARRAY, NULL),           /* Quaternion c param. */
        "quatern_d", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->quatern_d, 0, NPY_ARRAY_CARRAY, NULL),           /* Quaternion d param. */
        "qoffset_x", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->qoffset_x, 0, NPY_ARRAY_CARRAY, NULL),           /* Quaternion x shift. */
        "qoffset_y", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->qoffset_y, 0, NPY_ARRAY_CARRAY, NULL),           /* Quaternion y shift. */
        "qoffset_z", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, &nih->qoffset_z, 0, NPY_ARRAY_CARRAY, NULL),           /* Quaternion z shift. */
        "srow_x", PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT32, NULL, &nih->srow_x, 0, NPY_ARRAY_CARRAY, NULL),                 /* 1st row affine transform. */
        "srow_y", PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT32, NULL, &nih->srow_y, 0, NPY_ARRAY_CARRAY, NULL),                 /* 2nd row affine transform. */
        "srow_z", PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT32, NULL, &nih->srow_z, 0, NPY_ARRAY_CARRAY, NULL),                 /* 3rd row affine transform. */
        "intent_name", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, &nih->intent_name, 4, NPY_ARRAY_CARRAY, NULL),        /* 'name' or meaning of data. */
        "magic", PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, &nih->magic, 4, NPY_ARRAY_CARRAY, NULL)                     /* MUST be "ni1" or "n+1". */
    );
    return re;
}

static int nifti1_type_to_npy(int t_datatype)
{
    switch (t_datatype)
    {
    case DT_NONE:
        return NPY_VOID;
    case DT_BINARY:
        return NPY_VOID;
    case DT_UINT8:
        return NPY_UINT8;
    case DT_INT16:
        return NPY_INT16;
    case DT_INT32:
        return NPY_INT32;
    case DT_FLOAT32:
        return NPY_FLOAT32;
    case DT_COMPLEX64:
        return NPY_COMPLEX64;
    case DT_FLOAT64:
        return NPY_FLOAT64;
    case DT_RGB24:
        return NPY_UINT8; // todo
    case DT_ALL:
        return NPY_VOID;
    case DT_INT8:
        return NPY_INT8;
    case DT_UINT16:
        return NPY_UINT16;
    case DT_UINT32:
        return NPY_UINT32;
    case DT_INT64:
        return NPY_INT64;
    case DT_UINT64:
        return NPY_UINT64;
    case DT_FLOAT128:
        return NPY_FLOAT128;
    case DT_COMPLEX128:
        return NPY_COMPLEX128;
    case DT_COMPLEX256:
        return NPY_COMPLEX256;
    case DT_RGBA32:
        return NPY_UINT8; // todo
    default:
        return NPY_VOID;
    }
}

static PyObject *niftilib_read_volume_c(const PyObject *self, PyObject *args)
{
    char *param_filename = NULL;

    if (!PyArg_ParseTuple(args, "s", &param_filename))
        return NULL;

    nifti_image *nim = nifti_image_read(param_filename, 0);
    if (nim == NULL)
    {
        return NULL;
    }

    npy_intp dims[7] = {nim->dim[7], nim->dim[6], nim->dim[5], nim->dim[4], nim->dim[3], nim->dim[2], nim->dim[1]};

    PyObject *arr = PyArray_New(&PyArray_Type, nim->ndim, dims + 7 - nim->ndim, nifti1_type_to_npy(nim->datatype), NULL, NULL, 0, 1, NULL);

    nim->data = PyArray_DATA(arr);

    if (nifti_image_load(nim) != 0)
    {
        nim->data = NULL;
        nifti_image_free(nim);
        return NULL;
    };

    nim->data = NULL;

    nifti_image_free(nim);

    return arr;
}

static PyMethodDef niftilib_methods[] = {
    {
        "read_volume",
        niftilib_read_volume_c,
        METH_VARARGS,
        "Read nifti volume.",
    },
    {
        "read_header",
        niftilib_read_header_c,
        METH_VARARGS,
        "Read nifti header.",
    },
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef niftilib_definition = {
    PyModuleDef_HEAD_INIT,
    "nifti",
    "Reading and writing NIfTI files.",
    -1,
    niftilib_methods};

PyMODINIT_FUNC PyInit_nifti(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&niftilib_definition);
}
