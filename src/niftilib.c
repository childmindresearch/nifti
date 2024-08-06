#include <stdio.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <zlib.h>

#include "cnifti.h"

static PyObject *n1_header_to_raw_pydict(const cnifti_n1_header_t *nih)
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

// static PyObject *n2_header_to_raw_pydict(const cnifti_n2_header_t *nih)
// {
//     PyObject *h_sizeof_hdr = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);       /* MUST be 348 */
//     PyObject *h_data_type = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, NULL, 10, NPY_ARRAY_CARRAY, NULL);      /* ++UNUSED++ */
//     PyObject *h_db_name = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_STRING, NULL, NULL, 18, NPY_ARRAY_CARRAY, NULL);        /* ++UNUSED++ */
//     PyObject *h_extents = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);          /* ++UNUSED++ */
//     PyObject *h_session_error = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);    /* ++UNUSED++ */
//     PyObject *h_regular = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);           /* ++UNUSED++ */
//     PyObject *h_dim_info = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);          /* MRI slice ordering. */
//     PyObject *h_dim = PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_INT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);              /* Data array dimensions. */
//     PyObject *h_intent_p1 = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* 1st intent parameter. */
//     PyObject *h_intent_p2 = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* 2nd intent parameter. */
//     PyObject *h_intent_p3 = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* 3rd intent parameter. */
//     PyObject *h_intent_code = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* NIFTI_INTENT_* code. */
//     PyObject *h_datatype = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* Defines data type! */
//     PyObject *h_bitpix = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);           /* Number bits/voxel */
//     PyObject *h_slice_start = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* First slice index. */
//     PyObject *h_pixdim = PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* Grid spacings. */
//     PyObject *h_vox_offset = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);     /* Offset into .nii file */
//     PyObject *h_scl_slope = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Data scaling: slope. */
//     PyObject *h_scl_inter = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);      /* Data scaling: offset. */
//     PyObject *h_slice_end = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Last slice index. */
//     PyObject *h_slice_code = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Slice timing order. */
//     PyObject *h_xyzt_units = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_INT8, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Units of pixdim[1..4] */
//     PyObject *h_cal_max = PyArray_New(&PyArray_Type, 0, (npy_intp[]){1}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);        /* Max display intensity */

// }

static PyObject *n1_header_to_pydict(const cnifti_n1_header_t *nih)
{
    PyObject *h_sizeof_hdr = PyLong_FromLong(nih->sizeof_hdr);
    PyObject *h_data_type = PyUnicode_FromStringAndSize(nih->data_type, sizeof(nih->data_type));
    PyObject *h_db_name = PyUnicode_FromStringAndSize(nih->db_name, sizeof(nih->db_name));
    PyObject *h_extents = PyLong_FromLong(nih->extents);
    PyObject *h_session_error = PyLong_FromLong(nih->session_error);
    PyObject *h_regular = PyLong_FromLong(nih->regular);
    PyObject *h_dim_info = PyLong_FromLong(nih->dim_info);
    PyObject *h_dim = PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_INT16, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);              /* Data array dimensions. */
    PyObject *h_intent_p1 = PyFloat_FromDouble(nih->intent_p1);
    PyObject *h_intent_p2 = PyFloat_FromDouble(nih->intent_p2);
    PyObject *h_intent_p3 = PyFloat_FromDouble(nih->intent_p3);
    PyObject *h_intent_code = PyLong_FromLong(nih->intent_code);
    PyObject *h_datatype = PyLong_FromLong(nih->datatype);
    PyObject *h_bitpix = PyLong_FromLong(nih->bitpix);
    PyObject *h_slice_start = PyLong_FromLong(nih->slice_start);
    PyObject *h_pixdim = PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* Grid spacings. */
    PyObject *h_vox_offset = PyFloat_FromDouble(nih->vox_offset);
    PyObject *h_scl_slope = PyFloat_FromDouble(nih->scl_slope);
    PyObject *h_scl_inter = PyFloat_FromDouble(nih->scl_inter);
    PyObject *h_slice_end = PyLong_FromLong(nih->slice_end);
    PyObject *h_slice_code = PyLong_FromLong(nih->slice_code);
    PyObject *h_xyzt_units = PyLong_FromLong(nih->xyzt_units);
    PyObject *h_cal_max = PyFloat_FromDouble(nih->cal_max);
    PyObject *h_cal_min = PyFloat_FromDouble(nih->cal_min);
    PyObject *h_slice_duration = PyFloat_FromDouble(nih->slice_duration);
    PyObject *h_toffset = PyFloat_FromDouble(nih->toffset);
    PyObject *h_glmax = PyLong_FromLong(nih->glmax);
    PyObject *h_glmin = PyLong_FromLong(nih->glmin);
    PyObject *h_descrip = PyUnicode_FromStringAndSize(nih->descrip, sizeof(nih->descrip));
    PyObject *h_aux_file = PyUnicode_FromStringAndSize(nih->aux_file, sizeof(nih->aux_file));
    PyObject *h_qform_code = PyLong_FromLong(nih->qform_code);
    PyObject *h_sform_code = PyLong_FromLong(nih->sform_code);
    PyObject *h_quatern_b = PyFloat_FromDouble(nih->quatern_b);
    PyObject *h_quatern_c = PyFloat_FromDouble(nih->quatern_c);
    PyObject *h_quatern_d = PyFloat_FromDouble(nih->quatern_d);
    PyObject *h_qoffset_x = PyFloat_FromDouble(nih->qoffset_x);
    PyObject *h_qoffset_y = PyFloat_FromDouble(nih->qoffset_y);
    PyObject *h_qoffset_z = PyFloat_FromDouble(nih->qoffset_z);
    PyObject *h_srow_x = PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* 1st row affine transform. */
    PyObject *h_srow_y = PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* 2nd row affine transform. */
    PyObject *h_srow_z = PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* 3rd row affine transform. */
    PyObject *h_intent_name = PyUnicode_FromStringAndSize(nih->intent_name, sizeof(nih->intent_name));
    PyObject *h_magic = PyUnicode_FromStringAndSize(nih->magic, sizeof(nih->magic));

    memcpy(PyArray_DATA(h_dim), &nih->dim, sizeof(nih->dim));
    memcpy(PyArray_DATA(h_pixdim), &nih->pixdim, sizeof(nih->pixdim));
    memcpy(PyArray_DATA(h_srow_x), &nih->srow_x, sizeof(nih->srow_x));
    memcpy(PyArray_DATA(h_srow_y), &nih->srow_y, sizeof(nih->srow_y));
    memcpy(PyArray_DATA(h_srow_z), &nih->srow_z, sizeof(nih->srow_z));

    PyObject *re = Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,"
        "s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,"
        "s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,"
        "s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
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

static PyObject *n2_header_to_pydict(const cnifti_n1_header_t *nih)
{
    PyObject *h_sizeof_hdr = PyLong_FromLong(nih->sizeof_hdr);
    PyObject *h_data_type = PyUnicode_FromStringAndSize(nih->data_type, sizeof(nih->data_type));
    PyObject *h_db_name = PyUnicode_FromStringAndSize(nih->db_name, sizeof(nih->db_name));
    PyObject *h_extents = PyLong_FromLong(nih->extents);
    PyObject *h_session_error = PyLong_FromLong(nih->session_error);
    PyObject *h_regular = PyLong_FromLong(nih->regular);
    PyObject *h_dim_info = PyLong_FromLong(nih->dim_info);
    PyObject *h_dim = PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_INT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);              /* Data array dimensions. */
    PyObject *h_intent_p1 = PyFloat_FromDouble(nih->intent_p1);
    PyObject *h_intent_p2 = PyFloat_FromDouble(nih->intent_p2);
    PyObject *h_intent_p3 = PyFloat_FromDouble(nih->intent_p3);
    PyObject *h_intent_code = PyLong_FromLong(nih->intent_code);
    PyObject *h_datatype = PyLong_FromLong(nih->datatype);
    PyObject *h_bitpix = PyLong_FromLong(nih->bitpix);
    PyObject *h_slice_start = PyLong_FromLong(nih->slice_start);
    PyObject *h_pixdim = PyArray_New(&PyArray_Type, 1, (npy_intp[]){8}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* Grid spacings. */
    PyObject *h_vox_offset = PyFloat_FromDouble(nih->vox_offset);
    PyObject *h_scl_slope = PyFloat_FromDouble(nih->scl_slope);
    PyObject *h_scl_inter = PyFloat_FromDouble(nih->scl_inter);
    PyObject *h_slice_end = PyLong_FromLong(nih->slice_end);
    PyObject *h_slice_code = PyLong_FromLong(nih->slice_code);
    PyObject *h_xyzt_units = PyLong_FromLong(nih->xyzt_units);
    PyObject *h_cal_max = PyFloat_FromDouble(nih->cal_max);
    PyObject *h_cal_min = PyFloat_FromDouble(nih->cal_min);
    PyObject *h_slice_duration = PyFloat_FromDouble(nih->slice_duration);
    PyObject *h_toffset = PyFloat_FromDouble(nih->toffset);
    PyObject *h_glmax = PyLong_FromLong(nih->glmax);
    PyObject *h_glmin = PyLong_FromLong(nih->glmin);
    PyObject *h_descrip = PyUnicode_FromStringAndSize(nih->descrip, sizeof(nih->descrip));
    PyObject *h_aux_file = PyUnicode_FromStringAndSize(nih->aux_file, sizeof(nih->aux_file));
    PyObject *h_qform_code = PyLong_FromLong(nih->qform_code);
    PyObject *h_sform_code = PyLong_FromLong(nih->sform_code);
    PyObject *h_quatern_b = PyFloat_FromDouble(nih->quatern_b);
    PyObject *h_quatern_c = PyFloat_FromDouble(nih->quatern_c);
    PyObject *h_quatern_d = PyFloat_FromDouble(nih->quatern_d);
    PyObject *h_qoffset_x = PyFloat_FromDouble(nih->qoffset_x);
    PyObject *h_qoffset_y = PyFloat_FromDouble(nih->qoffset_y);
    PyObject *h_qoffset_z = PyFloat_FromDouble(nih->qoffset_z);
    PyObject *h_srow_x = PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* 1st row affine transform. */
    PyObject *h_srow_y = PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* 2nd row affine transform. */
    PyObject *h_srow_z = PyArray_New(&PyArray_Type, 1, (npy_intp[]){4}, NPY_FLOAT64, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);         /* 3rd row affine transform. */
    PyObject *h_intent_name = PyUnicode_FromStringAndSize(nih->intent_name, sizeof(nih->intent_name));
    PyObject *h_magic = PyUnicode_FromStringAndSize(nih->magic, sizeof(nih->magic));

    memcpy(PyArray_DATA(h_dim), &nih->dim, sizeof(nih->dim));
    memcpy(PyArray_DATA(h_pixdim), &nih->pixdim, sizeof(nih->pixdim));
    memcpy(PyArray_DATA(h_srow_x), &nih->srow_x, sizeof(nih->srow_x));
    memcpy(PyArray_DATA(h_srow_y), &nih->srow_y, sizeof(nih->srow_y));
    memcpy(PyArray_DATA(h_srow_z), &nih->srow_z, sizeof(nih->srow_z));

    PyObject *re = Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,"
        "s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,"
        "s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,"
        "s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
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

static int read_nifti_header(gzFile file_handle, cnifti_header_t *buf)
{
    if (gzread(file_handle, buf, sizeof(cnifti_n1_header_t)) < 0)
    {
        int err;
        printf("Error: Could not read nifti1 header %i '%s'\n", err, gzerror(file_handle, &err));
        return 0;
    }

    int32_t peek = cnifti_peek(*((uint32_t *)buf));

    if (peek == -1)
    {
        printf("Error: Not a valid nifti file\n");
        return 0;
    }
    if (peek & CNIFTI_PEEK_NIFTI2)
    {
        // Read the rest of the header
        if (gzread(file_handle, ((uint8_t *)buf) + sizeof(cnifti_n1_header_t), sizeof(cnifti_n2_header_t) - sizeof(cnifti_n1_header_t)) < 0)
        {
            printf("Error: Could not read nifti2 header\n");
            return 0;
        }
        cnifti_n2_header_t *header = &buf->n2_header;
        if (peek & CNIFTI_PEEK_SWAP)
        {
            cnifti_n2_header_swap(header);
        }
        return 2;
    }
    else
    {
        cnifti_n1_header_t *header = &buf->n1_header;
        if (peek & CNIFTI_PEEK_SWAP)
        {
            cnifti_n1_header_swap(header);
        }
        return 1;
    }
    return 0;
}

static PyObject *niftilib_read_header_c(const PyObject *self, PyObject *args)
{
    char *param_filename = NULL;

    if (!PyArg_ParseTuple(args, "s", &param_filename))
    {
        return NULL;
    }

    gzFile file_handle = gzopen(param_filename, "rb");

    if (file_handle == Z_NULL)
    {
        int err;
        printf("Error opening file %i: '%s'\n", errno, strerror(errno));
        PyErr_SetString(PyExc_IOError, "Error opening file.");
        return NULL;
    }

    cnifti_header_t nih;

    if (read_nifti_header(file_handle, &nih) != 1)
    {
        gzclose(file_handle);
        PyErr_SetString(PyExc_IOError, "Error reading NIfTI file.");
        return NULL;
    }

    gzclose(file_handle);
    
    PyObject *re = n1_header_to_raw_pydict(&nih.n1_header);
    return re;
}

static int nifti1_type_to_npy(int t_datatype)
{
    switch (t_datatype)
    {
    case CNIFTI_DT_UNKNOWN:
    case CNIFTI_DT_BINARY:
        return NPY_VOID;
    case CNIFTI_DT_UINT8:
        return NPY_UINT8;
    case CNIFTI_DT_INT16:
        return NPY_INT16;
    case CNIFTI_DT_INT32:
        return NPY_INT32;
    case CNIFTI_DT_FLOAT32:
        return NPY_FLOAT32;
    case CNIFTI_DT_COMPLEX64:
        return NPY_COMPLEX64;
    case CNIFTI_DT_FLOAT64:
        return NPY_FLOAT64;
#ifdef NPY_UINT24
    case DT_RGB24:
        return NPY_UINT24; // todo
#endif
    case CNIFTI_DT_INT8:
        return NPY_INT8;
    case CNIFTI_DT_UINT16:
        return NPY_UINT16;
    case CNIFTI_DT_UINT32:
        return NPY_UINT32;
    case CNIFTI_DT_INT64:
        return NPY_INT64;
    case CNIFTI_DT_UINT64:
        return NPY_UINT64;
#ifdef NPY_FLOAT128 // Not available on windows with MSVC
    case CNIFTI_DT_FLOAT128:
        return NPY_FLOAT128;
#endif
#ifdef NPY_COMPLEX128
    case CNIFTI_DT_COMPLEX128:
        return NPY_COMPLEX128;
#endif
#ifdef NPY_COMPLEX256 // Not available on windows with MSVC
    case CNIFTI_DT_COMPLEX256:
        return NPY_COMPLEX256;
#endif
    case CNIFTI_DT_RGBA32:
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

    gzFile file_handle = gzopen(param_filename, "rb");

    if (file_handle == Z_NULL)
    {
        int err;
        printf("Error opening file %i: '%s'\n", errno, strerror(errno));
        PyErr_SetString(PyExc_IOError, "Error opening file.");
        return NULL;
    }

    cnifti_header_t nih;

    if (read_nifti_header(file_handle, &nih) != 1)
    {
        gzclose(file_handle);
        PyErr_SetString(PyExc_IOError, "Error reading NIfTI file.");
        return NULL;
    }

    const cnifti_n1_header_t *header = &nih.n1_header;

    const npy_intp dims[7] = {header->dim[1], header->dim[2], header->dim[3], header->dim[4], header->dim[5], header->dim[6], header->dim[7]};

    const int16_t ndim = header->dim[0];

    PyObject *arr = PyArray_New(&PyArray_Type, ndim, dims, nifti1_type_to_npy(header->datatype), NULL, NULL, 0, 1, NULL);

    cnifti_extension_indicator_t ext_indicator;
    if (gzread(file_handle, &ext_indicator, sizeof(cnifti_extension_indicator_t)) < 0)
    {
        gzclose(file_handle);
        PyErr_SetString(PyExc_IOError, "Error: Could not read header extension\n");
        return NULL;
    }

    if (ext_indicator.has_extension)
    {
        // Read extension header
        cnifti_extension_header_t ext_header;
        if (gzread(file_handle, &ext_header, sizeof(cnifti_extension_header_t)) < 0)
        {
            gzclose(file_handle);
            PyErr_SetString(PyExc_IOError, "Error: Could not read extension header\n");
            return NULL;
        }

        // Skip extension data (TODO)
        if (gzseek(file_handle, ext_header.esize - 8, SEEK_CUR) < 0)
        {
            gzclose(file_handle);
            PyErr_SetString(PyExc_IOError, "Error: Could not skip extension data\n");
            return NULL;
        }
    }

    if (gzread(file_handle, PyArray_DATA(arr), cnifti_n1_header_array_size(header)) < 0)
    {
        gzclose(file_handle);
        PyErr_SetString(PyExc_IOError, "Error: Could not read data\n");
        return NULL;
    }

    gzclose(file_handle);
    return arr;
}

static PyObject *niftilib_read_extension_c(const PyObject *self, PyObject *args)
{
    char *param_filename = NULL;

    if (!PyArg_ParseTuple(args, "s", &param_filename))
    {
        return NULL;
    }

    gzFile file_handle = gzopen(param_filename, "rb");

    if (file_handle == Z_NULL)
    {
        int err;
        printf("Error opening file %i: '%s'\n", errno, strerror(errno));
        PyErr_SetString(PyExc_IOError, "Error opening file.");
        return NULL;
    }

    cnifti_header_t nih;

    if (read_nifti_header(file_handle, &nih) != 1)
    {
        gzclose(file_handle);
        PyErr_SetString(PyExc_IOError, "Error reading NIfTI file.");
        return NULL;
    }

    const cnifti_n1_header_t *header = &nih.n1_header;

    cnifti_extension_indicator_t ext_indicator;
    if (gzread(file_handle, &ext_indicator, sizeof(cnifti_extension_indicator_t)) < 0)
    {
        gzclose(file_handle);
        PyErr_SetString(PyExc_IOError, "Error: Could not read header extension\n");
        return NULL;
    }

    if (!ext_indicator.has_extension)
    {
        PyErr_SetString(PyExc_IOError, "Error: No extension present\n");
        return NULL;
    }

    // Read extension header
    cnifti_extension_header_t ext_header;
    if (gzread(file_handle, &ext_header, sizeof(cnifti_extension_header_t)) < 0)
    {
        gzclose(file_handle);
        PyErr_SetString(PyExc_IOError, "Error: Could not read extension header\n");
        return NULL;
    }

    // Read extension data
    PyObject *ext_data = PyArray_New(&PyArray_Type, 1, (npy_intp[]){ext_header.esize - 8}, NPY_UINT8, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    if (gzread(file_handle, PyArray_DATA(ext_data), ext_header.esize - 8) < 0)
    {
        gzclose(file_handle);
        PyErr_SetString(PyExc_IOError, "Error: Could not read extension data\n");
        return NULL;
    }

    gzclose(file_handle);

    // create a dictionary with the extension header and data
    const char *ecode_name = cnifti_ecode_name(ext_header.ecode);
    PyObject *ecode_name_py;
    if (ecode_name == NULL) {
        ecode_name_py = Py_None;
    } else {
        ecode_name_py = PyUnicode_FromStringAndSize(ecode_name, strlen(ecode_name));
    }
    PyObject *re = Py_BuildValue("{s:O,s:O,s:O,s:O}",
                                 "esize", PyLong_FromLong(ext_header.esize),
                                 "ecode", PyLong_FromLong(ext_header.ecode),
                                 "edata", ext_data,
                                 "ecode_name", ecode_name_py);
    
    return re;
}

/*typedef struct {
    PyObject_VAR_HEAD
    PyObject *field1;
    PyObject *field2;
} MyDataClassObject;

static PyTypeObject MyDataClassType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nifti.MyDataClass",
    .tp_basicsize = sizeof(MyDataClassObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};*/

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
    {
        "read_extension",
        niftilib_read_extension_c,
        METH_VARARGS,
        "Read nifti extension.",
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
    PyObject *m;
    //if (PyType_Ready(&MyDataClassType) < 0)
    //    return NULL;

    m = PyModule_Create(&niftilib_definition);
    if (m == NULL)
        return NULL;

    //Py_INCREF(&MyDataClassType);
    //PyModule_AddObject(m, "MyDataClass", (PyObject *)&MyDataClassType);

    Py_Initialize();
    import_array();
    return m;
}