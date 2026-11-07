//! Python-specific conversion for the owning carrier. No borrowed Python value
//! survives the call. Shape/bit fidelity is checked independently of artifacts.
use babeldoc_il_types::PythonValue as V;
use pyo3::{
    prelude::*,
    types::{
        PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString,
        PyTuple,
    },
};

pub fn from_python(v: &Bound<'_, PyAny>) -> PyResult<V> {
    let mut active = std::collections::HashSet::new();
    read(v, &mut active)
}
fn read(v: &Bound<'_, PyAny>, active: &mut std::collections::HashSet<usize>) -> PyResult<V> {
    if v.is_none() {
        return Ok(V::Null);
    }
    if v.is_instance_of::<PyBool>() {
        return Ok(V::Bool(v.extract()?));
    }
    if v.is_instance_of::<PyInt>() {
        // Avoid decimal str(int)'s configurable Python digit limit.
        let builtins = v.py().import("builtins")?;
        let hex: String = builtins.getattr("hex")?.call1((v,))?.extract()?;
        return Ok(V::Int(hex));
    }
    if v.is_instance_of::<PyFloat>() {
        return Ok(V::Float(v.extract()?));
    }
    if v.is_instance_of::<PyString>() {
        if let Ok(s) = v.extract::<String>() {
            return Ok(V::String(s));
        }
        let ord = v.py().import("builtins")?.getattr("ord")?;
        let points = v
            .try_iter()?
            .map(|c| ord.call1((c?,))?.extract::<u32>())
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(V::Unicode(points));
    }
    if let Ok(b) = v.cast::<PyBytes>() {
        return Ok(V::Bytes(b.as_bytes().to_vec()));
    }
    if v.is_instance_of::<PyByteArray>() {
        return Ok(V::ByteArray(
            v.py()
                .import("builtins")?
                .getattr("bytes")?
                .call1((v,))?
                .cast::<PyBytes>()?
                .as_bytes()
                .to_vec(),
        ));
    }
    let ptr = v.as_ptr() as usize;
    if !active.insert(ptr) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cyclic scalar carrier value",
        ));
    }
    let result = (|| {
        let items = |active: &mut std::collections::HashSet<usize>| {
            v.try_iter()?
                .map(|x| read(&x?, active))
                .collect::<PyResult<Vec<_>>>()
        };
        if v.is_instance_of::<PyList>() {
            return items(active).map(V::List);
        }
        if v.is_instance_of::<PyTuple>() {
            return items(active).map(V::Tuple);
        }
        if let Ok(d) = v.cast::<PyDict>() {
            return d
                .iter()
                .map(|(k, x)| Ok((read(&k, active)?, read(&x, active)?)))
                .collect::<PyResult<Vec<_>>>()
                .map(V::Dict);
        }
        if v.is_instance_of::<PySet>() {
            return items(active).map(V::Set);
        }
        if v.is_instance_of::<PyFrozenSet>() {
            return items(active).map(V::FrozenSet);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "unsupported scalar carrier type: {}",
            v.get_type().name()?
        )))
    })();
    active.remove(&ptr);
    result
}
pub fn to_python(py: Python<'_>, v: &V) -> PyResult<Py<PyAny>> {
    let builtins = py.import("builtins")?;
    Ok(match v {
        V::Null => py.None(),
        V::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        V::Int(s) => builtins.getattr("int")?.call1((s, 0))?.unbind(),
        V::Float(n) => n.into_pyobject(py)?.into_any().unbind(),
        V::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        V::Unicode(points) => {
            let chr = builtins.getattr("chr")?;
            let parts = PyList::new(
                py,
                points
                    .iter()
                    .map(|n| chr.call1((*n,)))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            "".into_pyobject(py)?
                .call_method1("join", (parts,))?
                .unbind()
        }
        V::List(a) | V::Tuple(a) | V::Set(a) | V::FrozenSet(a) => {
            let items = PyList::new(
                py,
                a.iter()
                    .map(|x| to_python(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            match v {
                V::List(_) => items.into_any().unbind(),
                V::Tuple(_) => builtins.getattr("tuple")?.call1((items,))?.unbind(),
                V::Set(_) => builtins.getattr("set")?.call1((items,))?.unbind(),
                _ => builtins.getattr("frozenset")?.call1((items,))?.unbind(),
            }
        }
        V::Dict(a) => {
            let d = PyDict::new(py);
            for (k, v) in a {
                d.set_item(to_python(py, k)?, to_python(py, v)?)?
            }
            d.into_any().unbind()
        }
        V::Bytes(b) => PyBytes::new(py, b).into_any().unbind(),
        V::ByteArray(b) => builtins
            .getattr("bytearray")?
            .call1((PyBytes::new(py, b),))?
            .unbind(),
    })
}
