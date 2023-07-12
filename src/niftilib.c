#include <stdio.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "nifti1_io.h"

static PyObject *nifti_header_to_pydict(const nifti_1_header *nih)
{
    PyObject *h_sizeof_hdr = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);       /* MUST be 348 */
    PyObject *h_data_type = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, NULL, 10, NPY_ARRAY_CARRAY, NULL);      /* ++UNUSED++ */
    PyObject *h_db_name = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, NULL, 18, NPY_ARRAY_CARRAY, NULL);        /* ++UNUSED++ */
    PyObject *h_extents = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);          /* ++UNUSED++ */
    PyObject *h_session_error = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);    /* ++UNUSED++ */
    PyObject *h_regular = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);           /* ++UNUSED++ */
    PyObject *h_dim_info = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);          /* MRI slice ordering. */
    PyObject *h_dim = PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);              /* Data array dimensions. */
    PyObject *h_intent_p1 = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* 1st intent parameter. */
    PyObject *h_intent_p2 = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* 2nd intent parameter. */
    PyObject *h_intent_p3 = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* 3rd intent parameter. */
    PyObject *h_intent_code = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* NIFTI_INTENT_* code. */
    PyObject *h_datatype = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* Defines data type! */
    PyObject *h_bitpix = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);           /* Number bits/voxel */
    PyObject *h_slice_start = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* First slice index. */
    PyObject *h_pixdim = PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* Grid spacings. */
    PyObject *h_vox_offset = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);     /* Offset into .nii file */
    PyObject *h_scl_slope = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Data scaling: slope. */
    PyObject *h_scl_inter = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Data scaling: offset. */
    PyObject *h_slice_end = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Last slice index. */
    PyObject *h_slice_code = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Slice timing order. */
    PyObject *h_xyzt_units = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Units of pixdim[1..4] */
    PyObject *h_cal_max = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Max display intensity */
    PyObject *h_cal_min = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Min display intensity */
    PyObject *h_slice_duration = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL); /* Time for 1 slice. */
    PyObject *h_toffset = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Time axis shift. */
    PyObject *h_glmax = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);            /* ++UNUSED++ */
    PyObject *h_glmin = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);            /* ++UNUSED++ */
    PyObject *h_descrip = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, NULL, 80, NPY_ARRAY_CARRAY, NULL);        /* any text you like. */
    PyObject *h_aux_file = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, NULL, 24, NPY_ARRAY_CARRAY, NULL);       /* auxiliary filename. */
    PyObject *h_qform_code = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);       /* NIFTI_XFORM_* code. */
    PyObject *h_sform_code = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);       /* NIFTI_XFORM_* code. */
    PyObject *h_quatern_b = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Quaternion b param. */
    PyObject *h_quatern_c = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Quaternion c param. */
    PyObject *h_quatern_d = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Quaternion d param. */
    PyObject *h_qoffset_x = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Quaternion x shift. */
    PyObject *h_qoffset_y = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Quaternion y shift. */
    PyObject *h_qoffset_z = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Quaternion z shift. */
    PyObject *h_srow_x = PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* 1st row affine transform. */
    PyObject *h_srow_y = PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* 2nd row affine transform. */
    PyObject *h_srow_z = PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* 3rd row affine transform. */
    PyObject *h_intent_name = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, NULL, 16, NPY_ARRAY_CARRAY, NULL);    /* 'name' or meaning of data. */
    PyObject *h_magic = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, NULL, 4, NPY_ARRAY_CARRAY, NULL);           /* MUST be "ni1" or "n+1". */

    memcpy(PyArray_DATA(h_sizeof_hdr), &nih->sizeof_hdr, sizeof(nih->sizeof_hdr));
    memcpy(PyArray_DATA(h_data_type), &nih->data_type, sizeof(nih->data_type));
    memcpy(PyArray_DATA(h_db_name), &nih->db_name, sizeof(nih->db_name));
    memcpy(PyArray_DATA(h_extents), &nih->extents, sizeof(nih->extents));
    memcpy(PyArray_DATA(h_session_error), &nih->session_error, sizeof(nih->session_error));
    memcpy(PyArray_DATA(h_regular), &nih->regular, sizeof(nih->regular));
    memcpy(PyArray_DATA(h_dim_info), &nih->dim_info, sizeof(nih->dim_info));
    memcpy(PyArray_DATA(h_dim), &nih->dim, sizeof(nih->dim));
    memcpy(PyArray_DATA(h_intent_p1), &nih->intent_p1, sizeof(nih->intent_p1));
    memcpy(PyArray_DATA(h_intent_p2), &nih->intent_p2, sizeof(nih->intent_p2));
    memcpy(PyArray_DATA(h_intent_p3), &nih->intent_p3, sizeof(nih->intent_p3));
    memcpy(PyArray_DATA(h_intent_code), &nih->intent_code, sizeof(nih->intent_code));
    memcpy(PyArray_DATA(h_datatype), &nih->datatype, sizeof(nih->datatype));
    memcpy(PyArray_DATA(h_bitpix), &nih->bitpix, sizeof(nih->bitpix));
    memcpy(PyArray_DATA(h_slice_start), &nih->slice_start, sizeof(nih->slice_start));
    memcpy(PyArray_DATA(h_pixdim), &nih->pixdim, sizeof(nih->pixdim));
    memcpy(PyArray_DATA(h_vox_offset), &nih->vox_offset, sizeof(nih->vox_offset));
    memcpy(PyArray_DATA(h_scl_slope), &nih->scl_slope, sizeof(nih->scl_slope));
    memcpy(PyArray_DATA(h_scl_inter), &nih->scl_inter, sizeof(nih->scl_inter));
    memcpy(PyArray_DATA(h_slice_end), &nih->slice_end, sizeof(nih->slice_end));
    memcpy(PyArray_DATA(h_slice_code), &nih->slice_code, sizeof(nih->slice_code));
    memcpy(PyArray_DATA(h_xyzt_units), &nih->xyzt_units, sizeof(nih->xyzt_units));
    memcpy(PyArray_DATA(h_cal_max), &nih->cal_max, sizeof(nih->cal_max));
    memcpy(PyArray_DATA(h_cal_min), &nih->cal_min, sizeof(nih->cal_min));
    memcpy(PyArray_DATA(h_slice_duration), &nih->slice_duration, sizeof(nih->slice_duration));
    memcpy(PyArray_DATA(h_toffset), &nih->toffset, sizeof(nih->toffset));
    memcpy(PyArray_DATA(h_glmax), &nih->glmax, sizeof(nih->glmax));
    memcpy(PyArray_DATA(h_glmin), &nih->glmin, sizeof(nih->glmin));
    memcpy(PyArray_DATA(h_descrip), &nih->descrip, sizeof(nih->descrip));
    memcpy(PyArray_DATA(h_aux_file), &nih->aux_file, sizeof(nih->aux_file));
    memcpy(PyArray_DATA(h_qform_code), &nih->qform_code, sizeof(nih->qform_code));
    memcpy(PyArray_DATA(h_sform_code), &nih->sform_code, sizeof(nih->sform_code));
    memcpy(PyArray_DATA(h_quatern_b), &nih->quatern_b, sizeof(nih->quatern_b));
    memcpy(PyArray_DATA(h_quatern_c), &nih->quatern_c, sizeof(nih->quatern_c));
    memcpy(PyArray_DATA(h_quatern_d), &nih->quatern_d, sizeof(nih->quatern_d));
    memcpy(PyArray_DATA(h_qoffset_x), &nih->qoffset_x, sizeof(nih->qoffset_x));
    memcpy(PyArray_DATA(h_qoffset_y), &nih->qoffset_y, sizeof(nih->qoffset_y));
    memcpy(PyArray_DATA(h_qoffset_z), &nih->qoffset_z, sizeof(nih->qoffset_z));
    memcpy(PyArray_DATA(h_srow_x), &nih->srow_x, sizeof(nih->srow_x));
    memcpy(PyArray_DATA(h_srow_y), &nih->srow_y, sizeof(nih->srow_y));
    memcpy(PyArray_DATA(h_srow_z), &nih->srow_z, sizeof(nih->srow_z));
    memcpy(PyArray_DATA(h_intent_name), &nih->intent_name, sizeof(nih->intent_name));
    memcpy(PyArray_DATA(h_magic), &nih->magic, sizeof(nih->magic));

    PyObject *re = Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
        "sizeof_hdr", h_sizeof_hdr,
        "data_type", h_data_type,
        "db_name", h_db_name,
        "extents", h_extents,
        "session_error", h_session_error,
        "regular", h_regular,
        "dim_info", h_dim_info,
        "dim", h_dim,
        "intent_p1", h_intent_p1,
        "intent_p2", h_intent_p2,
        "intent_p3", h_intent_p3,
        "intent_code", h_intent_code,
        "datatype", h_datatype,
        "bitpix", h_bitpix,
        "slice_start", h_slice_start,
        "pixdim", h_pixdim,
        "vox_offset", h_vox_offset,
        "scl_slope", h_scl_slope,
        "scl_inter", h_scl_inter,
        "slice_end", h_slice_end,
        "slice_code", h_slice_code,
        "xyzt_units", h_xyzt_units,
        "cal_max", h_cal_max,
        "cal_min", h_cal_min,
        "slice_duration", h_slice_duration,
        "toffset", h_toffset,
        "glmax", h_glmax,
        "glmin", h_glmin,
        "descrip", h_descrip,
        "aux_file", h_aux_file,
        "qform_code", h_qform_code,
        "sform_code", h_sform_code,
        "quatern_b", h_quatern_b,
        "quatern_c", h_quatern_c,
        "quatern_d", h_quatern_d,
        "qoffset_x", h_qoffset_x,
        "qoffset_y", h_qoffset_y,
        "qoffset_z", h_qoffset_z,
        "srow_x", h_srow_x,
        "srow_y", h_srow_y,
        "srow_z", h_srow_z,
        "intent_name", h_intent_name,
        "magic", h_magic);
    return re;
}

static PyObject *niftilib_read_header_c(const PyObject *self, PyObject *args)
{
    char *param_filename = NULL;

    if (!PyArg_ParseTuple(args, "s", &param_filename))
    {
        return NULL;
    }

    nifti_1_header *nih = nifti_read_header(param_filename, NULL, 0);
    if (nih == NULL)
    {
        PyErr_SetString(PyExc_IOError, "Error reading NIfTI file.");
        return NULL;
    }

    PyObject *re = nifti_header_to_pydict(nih);

    free(nih);
    
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
#ifdef NPY_UINT24
    case DT_RGB24:
        return NPY_UINT24; // todo
#endif
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
#ifdef NPY_FLOAT128 // Not available on windows with MSVC
    case DT_FLOAT128:
        return NPY_FLOAT128;
#endif
#ifdef NPY_COMPLEX128
    case DT_COMPLEX128:
        return NPY_COMPLEX128;
#endif
#ifdef NPY_COMPLEX256 // Not available on windows with MSVC
    case DT_COMPLEX256:
        return NPY_COMPLEX256;
#endif
    case DT_RGBA32:
        return NPY_UINT32; // todo
    default:
        return NPY_VOID;
    }
}

static PyObject *niftilib_read_volume_c(const PyObject *self, PyObject *args)
{
    char *param_filename = NULL;

    if (!PyArg_ParseTuple(args, "s", &param_filename))
    {
        return NULL;
    }

    nifti_image *nim = nifti_image_read(param_filename, 0);
    if (nim == NULL)
    {
        PyErr_SetString(PyExc_IOError, "Error reading NIfTI file.");
        return NULL;
    }

    npy_intp dims[7] = {nim->dim[7], nim->dim[6], nim->dim[5], nim->dim[4], nim->dim[3], nim->dim[2], nim->dim[1]};

    PyObject *arr = PyArray_New(&PyArray_Type, nim->ndim, dims + 7 - nim->ndim, nifti1_type_to_npy(nim->datatype), NULL, NULL, 0, 1, NULL);

    nim->data = PyArray_DATA(arr);

    if (nifti_image_load(nim) != 0)
    {
        nim->data = NULL;
        nifti_image_free(nim);
        PyErr_SetString(PyExc_IOError, "Error reading NIfTI file.");
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
