"""Generation backend for owned storage, borrowed navigation, and direct PyO3 traversal."""

from __future__ import annotations

import dataclasses
import types
import typing

from il_rust_m_declarations import OWNER_PRIORITIES
from il_rust_m_declarations import POOLS
from il_rust_m_declarations import TABLE_TYPES
from il_rust_m_declarations import VIEWS


def fields(cls, rust_type):
    hints = typing.get_type_hints(cls)
    for index, field in enumerate(dataclasses.fields(cls)):
        annotation = hints[field.name]
        origin = typing.get_origin(annotation)
        optional = origin in (typing.Union, types.UnionType)
        if optional:
            annotation = next(
                arg for arg in typing.get_args(annotation) if arg is not type(None)
            )
        many = typing.get_origin(annotation) is list
        if many:
            annotation = typing.get_args(annotation)[0]
        child = (
            annotation.__name__
            if isinstance(annotation, type) and dataclasses.is_dataclass(annotation)
            else None
        )
        scalar = rust_type(hints[field.name], owner=cls.__name__, field_name=field.name)
        if not child:
            scalar = scalar.replace("crate::", "babeldoc_il_types::")
            if (
                cls.__name__ == "GraphicState"
                and field.name == "passthrough_per_char_instruction"
            ):
                scalar = "Option<Instructions>"
        yield index, field.name, optional, many, child, scalar


def stored_scalar(cls, field, expression):
    if cls.__name__ == "PdfPath" and field in {"x", "y", "has_xy"}:
        return expression.replace(f"n.{field}", f"n.{field}()")
    return expression


def generate_tables(classes, rust_type):
    """Typed storage, and relocation of edges without changing object identity."""
    tables = [c for c in classes if c.__name__ in TABLE_TYPES]
    out = ["#[derive(Default)] struct TypedTables {"]
    out += [f"{c.__name__.lower()}:Vec<StoredNode<Data{c.__name__}>>," for c in tables]
    out += [
        "}",
        "impl TypedTables {fn shrink_to_fit(&mut self){",
        *[f"self.{c.__name__.lower()}.shrink_to_fit();" for c in tables],
        "} fn reserve_kind(&mut self,kind:Kind,count:usize){match kind{",
    ]
    out += [
        f"Kind::{c.__name__}=>self.{c.__name__.lower()}.reserve_exact(count),"
        for c in tables
    ]
    out += [
        "_=>unreachable!(),}} fn push(&mut self,node:Node)->usize{let Node{identity,data}=node;match data{"
    ]
    for cls in tables:
        name, field = cls.__name__, cls.__name__.lower()
        out += [
            f"NodeData::{name}(data)=>{{let i=self.{field}.len();self.{field}.push(StoredNode{{identity,data}});i}},"
        ]
    out += [
        '_=>unreachable!("only mutable schema objects enter typed tables"),',
        "}} fn node(&self,location:Location)->NodeRef<'_>{match location.kind{",
    ]
    for cls in tables:
        name = cls.__name__
        out += [
            f"Kind::{name}=>{{let n=&self.{name.lower()}[location.index];NodeRef{{identity:n.identity,data:NodeDataRef::{name}(&n.data)}}}},"
        ]
    out += [
        '_=>unreachable!("only table kinds have locations"),',
        "}} fn drain_kind(&mut self,kind:Kind,keys:&[usize],visit:&mut impl FnMut(usize,Node)){match kind{",
    ]
    for cls in tables:
        name = cls.__name__
        out += [
            f"Kind::{name}=>{{for (node,key) in std::mem::take(&mut self.{'r#box' if name == 'Box' else name.lower()}).into_iter().zip(keys){{visit(*key,Node{{identity:node.identity,data:NodeData::{name}(node.data)}});}}}},"
        ]
    out += ["_=>unreachable!(),}"]
    out += [
        "} fn reuse_kind(&mut self,source:&mut Self,kind:Kind,keys:&[usize],visit:&mut impl FnMut(usize,usize,&mut Node)){match kind{"
    ]
    for cls in tables:
        name = cls.__name__
        field = "r#box" if name == "Box" else name.lower()
        out += [
            f"Kind::{name}=>{{let mut values=std::mem::take(&mut source.{field});"
            f"for (index,(stored,key)) in values.iter_mut().zip(keys).enumerate(){{"
            f"let mut node=Node{{identity:stored.identity,data:NodeData::{name}(std::mem::take(&mut stored.data))}};"
            "visit(*key,index,&mut node);"
            f"let NodeData::{name}(data)=node.data else{{unreachable!()}};stored.data=data;"
            f"}}values.shrink_to_fit();self.{field}=values;}},"
        ]
    out += ["_=>unreachable!(),}"]
    out += ['} #[cfg(feature="memory-profile")] fn capacity_bytes(&self)->usize{']
    out += [
        "+".join(
            f"self.{c.__name__.lower()}.capacity()*std::mem::size_of::<StoredNode<Data{c.__name__}>>()"
            for c in tables
        ),
        "}}",
    ]
    # Most immutable path payloads cannot contain a table reference. Leave their
    # Arcs completely alone, including during compaction of large vector pages.
    relocatable = set(TABLE_TYPES)
    while True:
        following = relocatable | {
            c.__name__
            for c in classes
            if any(
                child in relocatable for _, _, _, _, child, _ in fields(c, rust_type)
            )
        }
        if following == relocatable:
            break
        relocatable = following
    immutable = [
        c for c in classes if c.__name__ in relocatable - TABLE_TYPES - {"Document"}
    ]
    out += [
        "impl Edge {fn table_edges(&self,visit:&mut impl FnMut(usize)){match self{",
        "Self::Shared(key)=>visit(*key),",
        "Self::Inline(n)=>{for e in n.data.edges(){e.table_edges(visit);}},",
    ]
    for cls in immutable:
        out += [f"Self::Immutable{cls.__name__}(n)=>{{"]
        for _, f, o, m, c, _ in fields(cls, rust_type):
            if c in relocatable:
                out += [
                    f"for e in &n.data.{f}{{e.table_edges(visit);}}"
                    if m
                    else f"if let Some(e)=&n.data.{f}{{e.table_edges(visit);}}"
                    if o
                    else f"n.data.{f}.table_edges(visit);"
                ]
        out += ["},"]
    out += [
        "_=>{}}} fn relocate(&mut self,indices:&Relocation,memo:&mut HashMap<u64,Edge>){match self{",
        "Self::Shared(key)=>{*key=indices.index(*key);debug_assert_ne!(*key,usize::MAX);},",
        'Self::Inline(_)=>unreachable!("compaction follows attachment"),',
    ]
    for cls in immutable:
        name = cls.__name__
        out += [
            f"Self::Immutable{name}(n)=>{{let identity=n.identity;if let Some(edge)=memo.get(&identity){{*self=edge.share();return;}}",
            f"let mut data=NodeData::{name}(Data{name}{{",
        ]
        for _, f, o, m, c, _ in fields(cls, rust_type):
            value = (
                f"n.data.{f}.iter().map(Edge::share).collect()"
                if c and m
                else f"n.data.{f}.as_ref().map(Edge::share)"
                if c and o
                else f"n.data.{f}.share()"
                if c
                else f"n.data.{f}.clone()"
            )
            out += [f"{f}:{value},"]
        out += [
            "});for e in data.edges_mut(){e.relocate(indices,memo);}",
            "let edge=Node{identity,data}.into_immutable_edge();memo.insert(identity,edge.share());*self=edge;},",
        ]
    out += ["_=>{}}}}"]
    return out


def generate_core(classes, rust_type):
    out = [
        "// Generated owned tree and document-table accessors from the IL schema.",
        "impl<P:Permission> TreeView<'_,P> {pub fn instruction_text(&self,value:&Option<Instructions>)->Result<Option<String>,AccessError> {render_instructions(value,&self.document_ref().history)}}",
        f"pub const CLASS_COUNT:usize={len(classes)};",
        "#[derive(Clone,Copy,Debug,Eq,PartialEq,Ord,PartialOrd,Hash)] pub enum Kind {",
        *[f"{c.__name__}," for c in classes],
        "}",
        "impl Kind {pub fn name(self)->&'static str {match self {",
        *[f'Self::{c.__name__}=>"{c.__name__}",' for c in classes],
        "}} pub fn is_pool(self)->bool {match self {",
        *[f"Self::{name}=>true," for name in sorted(POOLS)],
        "_=>false}} pub fn is_table(self)->bool {match self {",
        *[f"Self::{name}=>true," for name in sorted(TABLE_TYPES)],
        "_=>false}}}",
        "pub enum Edge {",
        "Inline(std::boxed::Box<Node>),Shared(usize),",
        *[
            f"Immutable{c.__name__}(Arc<ImmutableNode<Data{c.__name__}>>),"
            for c in classes
        ],
        "}",
        "pub mod types {use super::{Arc,Edge,ImmutableNode,Kind,SchemaType,Document as OwnedDocument,AccessError,ROOT};",
    ]
    for cls in classes:
        name = cls.__name__
        out += [
            f"#[derive(Debug)] pub enum {name} {{}}",
            (
                f"impl SchemaType for {name} {{const KIND:Kind=Kind::{name};"
                f'const NAME:&\'static str="{name}";type Data=super::Data{name};'
                f"fn from_immutable_edge(edge:&Edge)->Option<&Arc<ImmutableNode<Self::Data>>>"
                f"{{if let Edge::Immutable{name}(value)=edge{{Some(value)}}else{{None}}}}"
                f"fn into_immutable_edge(value:Arc<ImmutableNode<Self::Data>>)->Edge"
                f"{{Edge::Immutable{name}(value)}}"
            ),
        ]
        for mutable in (False, True):
            suffix = "_mut" if mutable else ""
            borrow = "&mut" if mutable else "&"
            if name == "Document":
                body = (
                    f"if index!=ROOT{{return Err(AccessError::WrongType)}};"
                    f"if let super::NodeData::Document(n)={borrow} doc.root.data"
                    "{Ok(n)}else{Err(AccessError::WrongType)}"
                )
            elif name in TABLE_TYPES:
                body = (
                    "let location=*doc.table.nodes.get(index).ok_or(AccessError::UnknownSlot)?;"
                    f"if location.kind!=Kind::{name}{{return Err(AccessError::WrongType)}};"
                    f"let tables=doc.table.scope{suffix}(location.scope);"
                    f"Ok({borrow} tables.{name.lower()}[location.index].data)"
                )
            else:
                body = "let _=(doc,index);Err(AccessError::WrongType)"
            out += [
                f"fn data{suffix}(doc:{borrow} OwnedDocument,index:usize)->Result<{borrow} Self::Data,AccessError>{{{body}}}"
            ]
        out += ["}"]
    out += [
        "}",
        "pub enum NodeData {",
        *[f"{c.__name__}(Data{c.__name__})," for c in classes],
        "}",
    ]

    def field_typ(optional, many, child, scalar, init=False):
        t = f"FieldValue<types::{child}>" if init else "Edge"
        return (
            (f"Vec<{t}>" if many else f"Option<{t}>" if optional else t)
            if child
            else scalar
        )

    def select(child, expr):
        t = "Cursor" if child in TABLE_TYPES else "Borrowed"
        return f"{t}::<types::{child}>::from_edge({expr}, generation)"

    for cls in classes:
        name = cls.__name__
        fs = list(fields(cls, rust_type))
        for init in (False, True):
            if name == "PdfPath" and not init:
                continue
            struct = f"{name}Init" if init else f"Data{name}"
            out += [f"#[derive(Default)] pub struct {struct} {{"]
            out += [
                f"pub {f}:{'CurveCtm' if name == 'PdfCurve' and f == 'ctm' and not init else field_typ(o, m, c, s, init)},"
                for _, f, o, m, c, s in fs
            ]
            out += ["}"]
        out += [
            f"impl {name}Init {{pub fn into_value(self)->FieldValue<types::{name}> {{#[allow(unused_mut)] let mut binding=Binding::Fresh;"
        ]
        for _, f, o, m, c, _ in fs:
            if c:
                ident = "r#box" if f == "box" else f
                out += [
                    f"for child in &self.{ident} {{binding.merge(child.binding);}}"
                    if m
                    else f"if let Some(child)=&self.{ident} {{binding.merge(child.binding);}}"
                    if o
                    else f"binding.merge(self.{ident}.binding);"
                ]
        if name == "PdfPath":
            out += [
                "FieldValue::new(NodeData::PdfPath(DataPdfPath::new(self.x,self.y,self.op.map(PathOp::from),self.has_xy)),binding)}}"
            ]
        else:
            out += [f"FieldValue::new(NodeData::{name}(Data{name} {{"]
        for _, f, o, m, c, _ in [] if name == "PdfPath" else fs:
            expr = (
                f"self.{f}.into_iter().map(|v|v.edge).collect()"
                if m
                else f"self.{f}.map(|v|v.edge)"
                if o
                else f"self.{f}.edge"
            )
            value = expr if c else "self." + f
            if name == "PdfCurve" and f == "ctm":
                value += ".into()"
            out += [f"{f}:{value},"]
        if name != "PdfPath":
            out += ["}),binding)}}"]
        table = name in TABLE_TYPES or name == "Document"
        out += [
            f"impl Cursor<types::{name}> {{"
            if table
            else f"#[allow(unused_variables)] impl<'n> Borrowed<'n,types::{name}> {{"
        ]
        life = "'a" if table else "'n"
        generic = "<'a,P:Permission>" if table else "<P:Permission>"
        view = "&'a TreeView<'_,P>" if table else "&TreeView<'_,P>"
        for _, f, o, m, c, s in fs:
            read = (
                "#[allow(unused_variables)] let generation=v.document_ref().generation;let n=self.data(v)?;"
                if table
                else "self.validate(v)?;#[allow(unused_variables)] let generation=v.document_ref().generation;let n=&self.node().data;"
            )
            if c:
                target = (
                    f"Cursor<types::{c}>"
                    if c in TABLE_TYPES
                    else f"Borrowed<{life},types::{c}>"
                )
                if m:
                    out += [
                        f"pub fn {f}_len<P:Permission>(&self,v:&TreeView<'_,P>)->Result<usize,AccessError>{{{read}Ok(n.{f}.len())}}",
                        f"pub fn {f}_at{generic}(&self,v:{view},i:usize)->Result<{target},AccessError>{{{read}{select(c, f'n.{f}.get(i).ok_or(AccessError::UnknownSlot)?')}}}",
                        f"pub fn {f}_iter{generic}(&self,v:{view})->Result<impl Iterator<Item=Result<{target},AccessError>>+{life},AccessError>{{{read}Ok(n.{f}.iter().map(move |edge|{select(c, 'edge')}))}}",
                    ]
                else:
                    result = f"Option<{target}>" if o else target
                    expr = (
                        f"n.{f}.as_ref().map(|edge|{select(c, 'edge')}).transpose()"
                        if o
                        else select(c, f"&n.{f}")
                    )
                    out += [
                        f"pub fn {f}{generic}(&self,v:{view})->Result<{result},AccessError>{{{read}{expr}}}"
                    ]
                if table:
                    typ = field_typ(o, m, c, s, True)
                    prepare = (
                        "values.into_iter().map(|c|Ok::<_,AccessError>(v.prepare(c)?)).collect::<Result<Vec<_>,_>>()?"
                        if m
                        else "values.map(|c|Ok::<_,AccessError>(v.prepare(c)?)).transpose()?"
                        if o
                        else "v.prepare(values)?"
                    )
                    out += [
                        f"pub fn set_{f}<P:Permission+Writes<types::{name}>>(&self,v:&mut TreeView<'_,P>,values:{typ})->Result<(),AccessError>{{self.validate(v)?;let prepared={prepare};let n=self.data_mut(v)?;n.{f}=prepared;Ok(())}}"
                    ]
                    if m:
                        out += [
                            f"pub fn push_{f}<P:Permission+Writes<types::{name}>>(&self,v:&mut TreeView<'_,P>,value:FieldValue<types::{c}>)->Result<(),AccessError>{{self.validate(v)?;let prepared=v.prepare(value)?;let n=self.data_mut(v)?;n.{f}.push(prepared);Ok(())}}"
                        ]
            else:
                copy = s in {
                    "f64",
                    "i64",
                    "bool",
                    "Option<f64>",
                    "Option<i64>",
                    "Option<bool>",
                }
                storage = (
                    "CurveCtm"
                    if name == "PdfCurve" and f == "ctm"
                    else "Option<PathOp>"
                    if name == "PdfPath" and f == "op"
                    else s
                )
                result = s if copy else f"&{life} {storage}"
                expr = stored_scalar(cls, f, f"n.{f}" if copy else f"&n.{f}")
                out += [
                    f"pub fn {f}{generic}(&self,v:{view})->Result<{result},AccessError>{{{read}Ok({expr})}}"
                ]
                if table:
                    value = (
                        "value.into()" if name == "PdfCurve" and f == "ctm" else "value"
                    )
                    out += [
                        f"pub fn set_{f}<P:Permission+Writes<types::{name}>>(&self,v:&mut TreeView<'_,P>,value:{s})->Result<(),AccessError>{{let n=self.data_mut(v)?;n.{f}={value};Ok(())}}"
                    ]
        if name == "GraphicState":
            out += [
                "pub fn instruction_text<P:Permission>(&self,v:&TreeView<'_,P>)->Result<Option<String>,AccessError>{render_instructions(self.passthrough_per_char_instruction(v)?,&v.document_ref().history)}"
            ]
        copy_read = (
            "#[allow(unused_variables)] let generation=v.document_ref().generation;let n=self.data(v)?;"
            if table
            else "self.validate(v)?;#[allow(unused_variables)] let generation=v.document_ref().generation;let n=&self.node().data;"
        )
        out += [
            f"pub fn copy_shallow<P:Permission>(&self,v:&mut TreeView<'_,P>)->Result<{name}Init,AccessError>{{{copy_read}Ok({name}Init {{"
        ]
        for _, f, o, m, c, _ in fs:
            if c:
                expr = (
                    f"n.{f}.iter().map(|e|FieldValue::from_edge(e.share(),generation)).collect()"
                    if m
                    else f"n.{f}.as_ref().map(|e|FieldValue::from_edge(e.share(),generation))"
                    if o
                    else f"FieldValue::from_edge(n.{f}.share(),generation)"
                )
            else:
                expr = stored_scalar(cls, f, f"n.{f}.clone()")
                if name == "PdfPath" and f == "op":
                    expr = "n.op.as_deref().map(str::to_owned)"
                elif name == "PdfCurve" and f == "ctm":
                    expr = "n.ctm.to_vec()"
            out += [f"{f}:{expr},"]
        out += ["})}}"]
    out += [
        f"pub type Immutable{c.__name__}=ImmutableNode<Data{c.__name__}>;"
        for c in classes
    ]
    out += [
        "#[allow(unused_variables)] impl NodeData {pub fn kind(&self)->Kind {match self {",
        *[f"Self::{c.__name__}(_)=>Kind::{c.__name__}," for c in classes],
        "}}",
    ]
    out += [
        "pub fn edges_mut(&mut self)->Vec<&mut Edge>{let mut out=vec![];match self {"
    ]
    for cls in classes:
        out += [f"Self::{cls.__name__}(n)=>{{"]
        for _, f, o, m, c, _ in fields(cls, rust_type):
            if c:
                out += [
                    f"out.extend(n.{f}.iter_mut());"
                    if m or o
                    else f"out.push(&mut n.{f});"
                ]
        out += ["},"]
    out += [
        "}out}",
        "pub fn edges(&self)->Vec<&Edge>{let mut out=vec![];match self {",
    ]
    for cls in classes:
        out += [f"Self::{cls.__name__}(n)=>{{"]
        for _, f, o, m, c, _ in fields(cls, rust_type):
            if c:
                out += [
                    f"out.extend(n.{f}.iter());" if m or o else f"out.push(&n.{f});"
                ]
        out += ["},"]
    out += [
        "}out}",
    ]
    readonly_start = len(out)
    out += [
        "fn dump(&self,d:&mut DumpContext<'_>,path:&str)->Result<Json,AccessError>{let mut out=serde_json::Map::new();match self {",
    ]
    for cls in classes:
        out += [f"Self::{cls.__name__}(n)=>{{"]
        for _, f, o, m, c, s in fields(cls, rust_type):
            if c:
                expr = (
                    f'Json::Array(n.{f}.iter().enumerate().map(|(i,c)|d.edge(c,format!("{{path}}.{f}[{{i}}]"))).collect::<Result<Vec<_>,_>>()?)'
                    if m
                    else f'n.{f}.as_ref().map(|c|d.edge(c,format!("{{path}}.{f}"))).transpose()?.unwrap_or(Json::Null)'
                    if o
                    else f'd.edge(&n.{f},format!("{{path}}.{f}"))?'
                )
            elif (
                cls.__name__ == "GraphicState"
                and f == "passthrough_per_char_instruction"
            ):
                expr = f'd.instructions(&n.{f},format!("{{path}}.{f}"))?'
            elif s == "Option<f64>":
                expr = f"n.{f}.map(float_artifact).unwrap_or(Json::Null)"
            elif s == "f64":
                expr = f"float_artifact(n.{f})"
            else:
                expr = (
                    f"serde_json::to_value(&n.{f}).map_err(|_|AccessError::WrongType)?"
                )
            expr = stored_scalar(cls, f, expr)
            out += [f'out.insert("{f}".into(),{expr});']
        out += ["},"]
    out += [
        "}Ok(Json::Object(out))}",
        '#[allow(unused_braces,unused_assignments,unused_variables)] fn write<W:std::io::Write>(&self,d:&mut DumpContext<\'_>,path:&str,out:&mut W)->Result<(),AccessError>{out.write_all(b"{").map_err(|_|AccessError::WrongType)?;let mut first=true;match self {',
    ]
    for cls in classes:
        out += [f"Self::{cls.__name__}(n)=>{{"]
        for _, f, o, m, c, s in sorted(
            fields(cls, rust_type), key=lambda item: item[1]
        ):
            ident = "r#box" if f == "box" else f
            if c:
                value = (
                    f'{{out.write_all(b"[").map_err(|_|AccessError::WrongType)?;for (i,e) in n.{ident}.iter().enumerate(){{if i>0{{out.write_all(b",").map_err(|_|AccessError::WrongType)?;}}d.write_edge(e,&format!("{{path}}.{f}[{{i}}]"),out)?;}}out.write_all(b"]").map_err(|_|AccessError::WrongType)?;}}'
                    if m
                    else f'{{if let Some(e)=&n.{ident}{{d.write_edge(e,&format!("{{path}}.{f}"),out)?;}}else{{out.write_all(b"null").map_err(|_|AccessError::WrongType)?;}}}}'
                    if o
                    else f'd.write_edge(&n.{ident},&format!("{{path}}.{f}"),out)?'
                )
            elif (
                cls.__name__ == "GraphicState"
                and f == "passthrough_per_char_instruction"
            ):
                value = f'{{let value=d.instructions(&n.{ident},format!("{{path}}.{f}"))?;serde_json::to_writer(&mut *out,&value).map_err(|_|AccessError::WrongType)?;}}'
            elif cls.__name__ == "BaseOperations" and f == "value":
                value = "{let value=decode_base_operations(&n.value)?;serde_json::to_writer(&mut *out,&value).map_err(|_|AccessError::WrongType)?;}"
            elif s in {"Option<f64>", "f64"}:
                v = (
                    f"n.{ident}.map(float_artifact).unwrap_or(Json::Null)"
                    if s.startswith("Option")
                    else f"float_artifact(n.{ident})"
                )
                value = f"{{let value={v};serde_json::to_writer(&mut *out,&value).map_err(|_|AccessError::WrongType)?;}}"
            else:
                value = f"{{serde_json::to_writer(&mut *out,&n.{ident}).map_err(|_|AccessError::WrongType)?;}}"
            value = stored_scalar(cls, f, value)
            out += [
                f'if !first{{out.write_all(b",").map_err(|_|AccessError::WrongType)?;}}first=false;out.write_all(b"\\"{f}\\":").map_err(|_|AccessError::WrongType)?;{value}'
            ]
        out += ["},"]
    out += [
        '}out.write_all(b"}").map_err(|_|AccessError::WrongType)?;Ok(())}',
        "#[allow(unused_variables)] fn discover(&self,d:&mut DumpContext<'_>,path:&str)->Result<(),AccessError>{match self {",
    ]
    for cls in classes:
        out += [f"Self::{cls.__name__}(n)=>{{"]
        for _, f, o, m, c, _s in fields(cls, rust_type):
            ident = "r#box" if f == "box" else f
            if c:
                if m:
                    expr = f'for (i,e) in n.{ident}.iter().enumerate(){{d.discover_edge(e,&format!("{{path}}.{f}[{{i}}]"))?;}}'
                elif o:
                    expr = f'if let Some(e)=&n.{ident}{{d.discover_edge(e,&format!("{{path}}.{f}"))?;}}'
                else:
                    expr = f'd.discover_edge(&n.{ident},&format!("{{path}}.{f}"))?;'
            elif (
                cls.__name__ == "GraphicState"
                and f == "passthrough_per_char_instruction"
            ):
                expr = f'd.instructions(&n.{ident},format!("{{path}}.{f}"))?;'
            else:
                continue
            out += [expr]
        out += ["},"]
    out += ["}Ok(())}", "pub fn payload_utf8_bytes(&self)->usize {match self {"]
    for cls in classes:
        expr = []
        for _, f, _, _, c, s in fields(cls, rust_type):
            if not c:
                expr.append(
                    f"n.{f}.utf8_bytes()"
                    if cls.__name__ == "PdfCurve" and f == "ctm"
                    else f"n.{f}.iter().map(PythonValue::utf8_bytes).sum::<usize>()"
                    if s == "Vec<babeldoc_il_types::PythonValue>"
                    else f"instruction_utf8_bytes(&n.{f})"
                    if cls.__name__ == "GraphicState"
                    and f == "passthrough_per_char_instruction"
                    else stored_scalar(
                        cls,
                        f,
                        f"json_utf8_bytes(&serde_json::to_value(&n.{f}).unwrap_or(Json::Null))",
                    )
                )
        out += [f"Self::{cls.__name__}(n)=>" + "+".join(expr or ["0"]) + ","]
    out += ["}}}"]
    # Borrowed payload dispatch uses the identical serializers. Storage changes do
    # not introduce another artifact implementation or construct owning Nodes.
    readonly = out[readonly_start:]
    out[readonly_start:] = ["}"]
    out += [
        "pub enum NodeDataRef<'a> {",
        *[f"{c.__name__}(&'a Data{c.__name__})," for c in classes],
        "}",
        "impl NodeData {fn as_ref(&self)->NodeDataRef<'_>{match self{",
        *[f"Self::{c.__name__}(n)=>NodeDataRef::{c.__name__}(n)," for c in classes],
        "}}}",
        "#[allow(unused_variables)] impl NodeDataRef<'_> {fn kind(&self)->Kind{match self{",
        *[f"Self::{c.__name__}(_)=>Kind::{c.__name__}," for c in classes],
        "}} fn edges(&self)->Vec<&Edge>{let mut out=vec![];match self{",
    ]
    for cls in classes:
        out += [f"Self::{cls.__name__}(n)=>{{"]
        for _, f, o, m, c, _ in fields(cls, rust_type):
            if c:
                out += [
                    f"out.extend(n.{f}.iter());" if m or o else f"out.push(&n.{f});"
                ]
        out += ["},"]
    out += ["}out}", *readonly]
    out += generate_tables(classes, rust_type)
    immutable = [
        cls
        for cls in classes
        if cls.__name__ not in TABLE_TYPES and cls.__name__ != "Document"
    ]
    for cls in immutable:
        name = cls.__name__
        out += [
            f"#[allow(unused_variables)] fn dump_data_{name.lower()}(n:&Data{name},d:&mut DumpContext<'_>,path:&str)->Result<Json,AccessError>{{let mut out=serde_json::Map::new();"
        ]
        for _, f, o, m, c, s in fields(cls, rust_type):
            if c:
                expr = (
                    f'Json::Array(n.{f}.iter().enumerate().map(|(i,c)|d.edge(c,format!("{{path}}.{f}[{{i}}]"))).collect::<Result<Vec<_>,_>>()?)'
                    if m
                    else f'n.{f}.as_ref().map(|c|d.edge(c,format!("{{path}}.{f}"))).transpose()?.unwrap_or(Json::Null)'
                    if o
                    else f'd.edge(&n.{f},format!("{{path}}.{f}"))?'
                )
            elif s == "Option<f64>":
                expr = f"n.{f}.map(float_artifact).unwrap_or(Json::Null)"
            elif s == "f64":
                expr = f"float_artifact(n.{f})"
            else:
                expr = (
                    f"serde_json::to_value(&n.{f}).map_err(|_|AccessError::WrongType)?"
                )
            expr = stored_scalar(cls, f, expr)
            out += [f'out.insert("{f}".into(),{expr});']
        out += [
            "Ok(Json::Object(out))}",
            "#[allow(unused_braces,unused_assignments,unused_variables)] fn write_data_"
            + name.lower()
            + "<W:std::io::Write>(n:&Data"
            + name
            + ',d:&mut DumpContext<\'_>,path:&str,out:&mut W)->Result<(),AccessError>{out.write_all(b"{").map_err(|_|AccessError::WrongType)?;let mut first=true;',
        ]
        for _, f, o, m, c, s in sorted(
            fields(cls, rust_type), key=lambda item: item[1]
        ):
            ident = "r#box" if f == "box" else f
            if c:
                value = (
                    f'{{out.write_all(b"[").map_err(|_|AccessError::WrongType)?;for (i,e) in n.{ident}.iter().enumerate(){{if i>0{{out.write_all(b",").map_err(|_|AccessError::WrongType)?;}}d.write_edge(e,&format!("{{path}}.{f}[{{i}}]"),out)?;}}out.write_all(b"]").map_err(|_|AccessError::WrongType)?;}}'
                    if m
                    else f'{{if let Some(e)=&n.{ident}{{d.write_edge(e,&format!("{{path}}.{f}"),out)?;}}else{{out.write_all(b"null").map_err(|_|AccessError::WrongType)?;}}}}'
                    if o
                    else f'd.write_edge(&n.{ident},&format!("{{path}}.{f}"),out)?'
                )
            elif s in {"Option<f64>", "f64"}:
                v = (
                    f"n.{ident}.map(float_artifact).unwrap_or(Json::Null)"
                    if s.startswith("Option")
                    else f"float_artifact(n.{ident})"
                )
                value = f"{{let value={v};serde_json::to_writer(&mut *out,&value).map_err(|_|AccessError::WrongType)?;}}"
            elif cls.__name__ == "BaseOperations" and f == "value":
                value = "{let value=decode_base_operations(&n.value)?;serde_json::to_writer(&mut *out,&value).map_err(|_|AccessError::WrongType)?;}"
            else:
                value = f"{{serde_json::to_writer(&mut *out,&n.{ident}).map_err(|_|AccessError::WrongType)?;}}"
            value = stored_scalar(cls, f, value)
            out += [
                f'if !first{{out.write_all(b",").map_err(|_|AccessError::WrongType)?;}}first=false;out.write_all(b"\\"{f}\\":").map_err(|_|AccessError::WrongType)?;{value}'
            ]
        out += ['out.write_all(b"}").map_err(|_|AccessError::WrongType)?;Ok(())}']
        out += [
            f"#[allow(unused_variables)] fn discover_data_{name.lower()}(n:&Data{name},d:&mut DumpContext<'_>,path:&str)->Result<(),AccessError>{{"
        ]
        for _, f, o, m, c, _s in fields(cls, rust_type):
            ident = "r#box" if f == "box" else f
            if c:
                if m:
                    expr = f'for (i,e) in n.{ident}.iter().enumerate(){{d.discover_edge(e,&format!("{{path}}.{f}[{{i}}]"))?;}}'
                elif o:
                    expr = f'if let Some(e)=&n.{ident}{{d.discover_edge(e,&format!("{{path}}.{f}"))?;}}'
                else:
                    expr = f'd.discover_edge(&n.{ident},&format!("{{path}}.{f}"))?;'
            elif (
                cls.__name__ == "GraphicState"
                and f == "passthrough_per_char_instruction"
            ):
                expr = f'd.instructions(&n.{ident},format!("{{path}}.{f}"))?;'
            else:
                continue
            out += [expr]
        out += ["Ok(())}"]
        expr = []
        for _, f, o, m, c, s in fields(cls, rust_type):
            if c:
                iterator = f"n.{f}.iter()" if o or m else f"std::iter::once(&n.{f})"
                expr.append(
                    f"{iterator}.map(|edge|edge.payload_utf8_bytes(seen)).sum::<usize>()"
                )
            else:
                expr.append(
                    f"n.{f}.utf8_bytes()"
                    if cls.__name__ == "PdfCurve" and f == "ctm"
                    else f"n.{f}.iter().map(PythonValue::utf8_bytes).sum::<usize>()"
                    if s == "Vec<babeldoc_il_types::PythonValue>"
                    else stored_scalar(
                        cls,
                        f,
                        f"json_utf8_bytes(&serde_json::to_value(&n.{f}).unwrap_or(Json::Null))",
                    )
                )
        out += [
            f"#[allow(unused_variables)] fn payload_data_{name.lower()}(n:&Data{name},seen:&mut HashSet<u64>)->usize{{"
            + "+".join(expr or ["0"])
            + "}"
        ]
    out += [
        "impl Edge {pub fn payload_utf8_bytes(&self,seen:&mut HashSet<u64>)->usize{match self{",
        "Self::Inline(_)|Self::Shared(_)=>0,",
        *[
            (
                f"Self::Immutable{cls.__name__}(node)=>"
                "if !seen.insert(node.identity){0}"
                f"else{{payload_data_{cls.__name__.lower()}(&node.data,seen)}},"
            )
            for cls in immutable
        ],
        '_=>unreachable!("table types are always shared"),',
        "}}}",
        "fn share_edge(edge:&Edge)->Edge{match edge{",
        'Edge::Inline(_)=>unreachable!("only attached values may be shared"),',
        "Edge::Shared(index)=>Edge::Shared(*index),",
        *[
            f"Edge::Immutable{c.__name__}(node)=>Edge::Immutable{c.__name__}(Arc::clone(node)),"
            for c in classes
        ],
        "}}",
        "fn dump_immutable_edge(d:&mut DumpContext<'_>,edge:&Edge,path:&str)->Result<Json,AccessError>{match edge{",
        *[
            (
                f"Edge::Immutable{cls.__name__}(node)=>{{"
                f"d.alias(Kind::{cls.__name__},node.identity,path);"
                f"dump_data_{cls.__name__.lower()}(&node.data,d,path)}}"
            )
            for cls in immutable
        ],
        '_=>unreachable!("table types are always shared"),',
        "}}",
        "fn write_immutable_edge<W:std::io::Write>(d:&mut DumpContext<'_>,edge:&Edge,path:&str,out:&mut W)->Result<(),AccessError>{match edge{",
        *[
            (
                f"Edge::Immutable{cls.__name__}(node)=>{{d.alias(Kind::{cls.__name__},node.identity,path);write_data_{cls.__name__.lower()}(&node.data,d,path,out)}}"
            )
            for cls in immutable
        ],
        '_=>unreachable!("table types are always shared"),',
        "}}",
        "fn discover_immutable_edge(d:&mut DumpContext<'_>,edge:&Edge,path:&str)->Result<(),AccessError>{match edge{",
        *[
            (
                f"Edge::Immutable{cls.__name__}(node)=>{{d.alias(Kind::{cls.__name__},node.identity,path);discover_data_{cls.__name__.lower()}(&node.data,d,path)}}"
            )
            for cls in immutable
        ],
        '_=>unreachable!("table types are always shared"),',
        "}}",
        "impl Node {fn into_immutable_edge(self)->Edge{let Self{identity,data}=self;match data{",
        *[
            (
                f"NodeData::{c.__name__}(data)=>Edge::Immutable{c.__name__}("
                f"Arc::new(ImmutableNode{{identity,data}})),"
            )
            for c in immutable
        ],
        '_=>unreachable!("table and root nodes stay in NodeData"),',
        "}}}",
    ]
    for name, declaration in VIEWS.items():
        write = (
            {c.__name__ for c in classes}
            if declaration["write"] == "all"
            else set(declaration["write"])
        )
        values = ",".join(str(c.__name__ in write).lower() for c in classes)
        out += [
            f"pub enum {name} {{}}",
            f"impl Permission for {name} {{const WRITE:[bool;CLASS_COUNT]=[{values}];}}",
        ]
        out += [f"impl Writes<types::{kind}> for {name} {{}}" for kind in sorted(write)]
        narrow = declaration["narrow"]
        if narrow:
            out += [
                f"impl TreeView<'_,{name}> {{pub fn as_{narrow.lower()}(&mut self)->TreeView<'_,{narrow}>{{TreeView{{doc:self.doc.reborrow(),_permission:PhantomData}}}}}}"
            ]
    import re

    return re.sub(r"(?<![\w#])box(?=[:(.,<=\[{])", "r#box", "\n".join(out) + "\n")


def generate_reader(classes, rust_type):
    out = [
        "// Generated direct object-graph scan and value loading; Python memo is loader-local.",
        "use pyo3::types::PyAnyMethods;",
    ]
    out += [
        "fn scan_node(value:&Bound<'_,PyAny>,c:&mut MBuildContext,context:ScanContext,path:String,priority:i32)->PyResult<()> {",
        "let name=value.get_type().name()?.to_str()?.to_owned();match name.as_str() {",
    ]
    for cls in classes:
        name = cls.__name__
        out += [f'"{name}"=>scan_{name.lower()}(value,c,context,path,priority),']
    out += [
        '_=>Err(PyTypeError::new_err(format!("unknown schema class {name}"))),',
        "}}",
    ]
    for cls in classes:
        name = cls.__name__
        out += [
            f"fn scan_{name.lower()}(value:&Bound<'_,PyAny>,c:&mut MBuildContext,context:ScanContext,path:String,priority:i32)->PyResult<()> {{",
            f"let (context,first)=c.observe(value,Kind::{name},context,&path,priority,Kind::{name}.is_pool())?;let _=context;if !first {{return Ok(())}};",
        ]
        for _, f, optional, many, child, _ in fields(cls, rust_type):
            if not child:
                continue
            priority = OWNER_PRIORITIES.get(f"{name}.{f}", 0)
            out += [f'let child=value.getattr("{f}")?;']
            if many:
                out += [
                    f'for (i,item) in child.try_iter()?.enumerate() {{scan_{child.lower()}(&item?,c,context,format!("{{path}}.{f}[{{i}}]"),{priority})?;}}'
                ]
            elif optional:
                out += [
                    f'if !child.is_none() {{scan_{child.lower()}(&child,c,context,format!("{{path}}.{f}"),{priority})?;}}'
                ]
            else:
                out += [
                    f'scan_{child.lower()}(&child,c,context,format!("{{path}}.{f}"),{priority})?;'
                ]
        out += ["Ok(())}"]
    out += [
        "fn load_data(value:&Bound<'_,PyAny>,c:&mut MBuildContext,path:&str)->PyResult<NodeData> {",
        "let name=value.get_type().name()?.to_str()?.to_owned();match name.as_str() {",
    ]
    for cls in classes:
        name = cls.__name__
        if name == "PdfPath":
            out += [
                '"PdfPath"=>Ok(NodeData::PdfPath(DataPdfPath::new(',
                '<Option<f64> as crate::FromPyValue>::from_py(&value.getattr("x")?,&mut c.scalar_context)?,',
                '<Option<f64> as crate::FromPyValue>::from_py(&value.getattr("y")?,&mut c.scalar_context)?,',
                '<Option<String> as crate::FromPyValue>::from_py(&value.getattr("op")?,&mut c.scalar_context)?.map(ir::PathOp::from),',
                '<Option<bool> as crate::FromPyValue>::from_py(&value.getattr("has_xy")?,&mut c.scalar_context)?))),',
            ]
            continue
        out += [f'"{name}"=>Ok(NodeData::{name}(Data{name} {{']
        for _, f, optional, many, child, scalar in fields(cls, rust_type):
            if child:
                if many:
                    expr = (
                        f'c.load_list(&value.getattr("{f}")?,&format!("{{path}}.{f}"))?'
                    )
                elif optional:
                    expr = f'c.load_optional(&value.getattr("{f}")?,&format!("{{path}}.{f}"))?'
                else:
                    expr = f'c.load(&value.getattr("{f}")?,format!("{{path}}.{f}"))?'
            elif (
                cls.__name__ == "GraphicState"
                and f == "passthrough_per_char_instruction"
            ):
                expr = f'c.instructions(&value.getattr("{f}")?)?'
            else:
                expr = f'<{scalar} as crate::FromPyValue>::from_py(&value.getattr("{f}")?,&mut c.scalar_context)?'
            if name == "PdfCurve" and f == "ctm":
                expr = f"({expr}).into()"
            out += [f"{f}:{expr},"]
        out += ["})),"]
    out += [
        '_=>Err(PyTypeError::new_err(format!("unknown schema class {name}"))),',
        "}}",
    ]
    result = "\n".join(out) + "\n"
    import re

    return re.sub(r"(?<![\w#])box(?=:)", "r#box", result)
