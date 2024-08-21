from enum import Enum
from typing import Callable
from typing import Optional
from dataclasses import dataclass
import os
import re
import io
import argparse

class MetaType(Enum):
    Package = 0
    Class = 1
    Enum = 2
    Module = 3
    Function = 4

name_to_meta_type = {
    'pypackage': MetaType.Package,
    'pyclass': MetaType.Class,
    'pyenum': MetaType.Enum,
    'pymodule': MetaType.Module,

    'hompackage': MetaType.Package,
    'hommodule': MetaType.Module,
    'homclass': MetaType.Class,
    'homfunction': MetaType.Function,
}

link_re = re.compile(r'\[(\w+:)?(?P<name>[\w\.:]+)(#\w+)?\]')
link_in_quote_re = re.compile(r'\[\w+:(?P<name>[\w\.:]+)(#\w+)?\]')
optinal_param_re = re.compile(r' \[(?P<name>\w+)\]')
py_type_re = re.compile(r'[a-zA-Z_]\w*\.[\w\.:]+')

def get_typo_fix_function():
    typo_word_list = [
        'true', 'True',
        'false', 'False',
        'string', 'str',
        'obj', 'object',
        'Bool', 'bool',
        'double', 'float',
        'floats', 'float',
        'any python object', 'Any',
        'any python obect', 'Any',
        'strings', 'str',
        'children=True editables=True', 'children=True, editables=True',
        'Nodes', 'Node',
        'Vector3s', 'Vector3',
        'tuples', 'tuple',
        'Tracks', 'Track',
        'long', 'int',
        'ints', 'int',
        'integer', 'int',
        'dictionary', 'dict',
        'HOM_Geometry', 'Geometry',
        'enum values', '',
        'enum value', '',
        'houNode', 'hou.Node',
    ]
    # make typo_list into pairs
    typo_re_pair_list = [(re.compile(r'\b' + typo_word_list[i] + r'\b'), typo_word_list[i + 1]) for i in range(0, len(typo_word_list), 2)]
    
    typo_replace_list = [
        '`str` for Python 2, `bytes` for Python 3', 'bytes',
        '`dict` mapping', 'dict of',
    ]
    type_replace_pair_list = [(typo_replace_list[i], typo_replace_list[i + 1]) for i in range(0, len(typo_replace_list), 2)]
    
    def typo_fix(s: str):
        for typo_re, replace_str in typo_re_pair_list:
            s = typo_re.sub(replace_str, s)
        
        for typo_str, replace_str in type_replace_pair_list:
            s = s.replace(typo_str, replace_str)

        return s
    return typo_fix
typo_fix = get_typo_fix_function()

def repair_optional_param_str(m: re.Match):
    name = m.group('name')
    return ' ' + name + '=None'

def build_container_type_str(parts: list[str]) -> str:
    if len(parts) <= 2:
        return ' | '.join(parts)
    head0 = parts[0]
    head1 = parts[1]
    if head1 == 'of':
        tail = parts[2:]
        if len(tail) >= 4 and tail[2] == 'of':
            # tuple of int tuple of float
            type_str_0 = build_container_type_str(parts[:3])
            type_str_1 = build_container_type_str(parts[3:])
            return type_str_0 + ' | ' + type_str_1
        if head0 == 'dict':
            if len(tail) >= 3 and tail[1] == 'to':
                value_type_str = build_container_type_str(tail[2:])
                return f'dict[{tail[0]}, {value_type_str}]'
            else:
                value_type_str = build_container_type_str(tail)
                return f'dict[str, {value_type_str}]'
        elem_type_str = build_container_type_str(tail)
        if head0 == 'list':
            return f'list[{elem_type_str}]'
        if head0 == 'generator':
            return f'Generator[{elem_type_str}, None, None]'
        if head0 == 'iterable':
            return f'Iterable[{elem_type_str}]'
        if head0 == 'tuple':
            if len(tail) == 3 and tail[2] == 'tuple':
                # tuple of int string tuple
                return f'tuple[tuple[{tail[0]} | {tail[1]}], ...]'
            else:
                return f'tuple[{elem_type_str}, ...]'
        if head0.endswith('-tuple'):
            tuple_size = int(head0[:-len('-tuple')])
            inner_str = ', '.join([elem_type_str] * tuple_size)
            return f'tuple[{inner_str}]'
        raise NotImplemented
    elif head0 == 'dict' and len(parts) >= 4 and parts[2] == 'to':
        # dict str to int
        value_type_str = build_container_type_str(parts[3:])
        return f'dict[{parts[1]}, {value_type_str}]'
    else:
        return head0 + ' | ' + build_container_type_str(parts[1:])

def repair_type_str(s: str) -> str:
    if s.strip() == '':
        return ''
    
    if s.strip() == 'Sequence[OpNode, int | str, int | str]]':
        return 'Sequence[tuple[OpNode, int | str, int | str]]'
    if s.startswith('tuple of (') and s.endswith(')'):
        inner_s = s[len('tuple of ('):-len(')')]
        inner_type_str_list = [repair_type_str(s1) for s1 in inner_s.split(',')]
        return f'tuple[{", ".join(inner_type_str_list)}]'
    if s.startswith('dict of (') and s.endswith(') pairs'):
        inner_s = s[len('dict of ('):-len(') pairs')]
        inner_type_str_list = [repair_type_str(s1) for s1 in inner_s.split(',')]
        return f'dict[{inner_type_str_list[0]}, {inner_type_str_list[1]}]'
    if s.startswith('dict of (') and ') to ' in s:
        to_index = s.index(') to ')
        innser_s = s[len('dict of ('):to_index]
        inner_type_str_list = [repair_type_str(s1) for s1 in innser_s.split(',')]
        tuple_type_params = ', '.join(inner_type_str_list)
        value_type_s = repair_type_str(s[to_index + len(') to '):])
        return f'dict[tuple[{tuple_type_params}], {value_type_s}]'
    if s.startswith('dict of ') and ' to ' in s:
        to_index = s.index(' to ')
        key_type_s = repair_type_str(s[len('dict of '):to_index])
        value_type_s = repair_type_str(s[to_index + len(' to '):])
        return f'dict[{key_type_s}, {value_type_s}]'
    if s.startswith('(') and s.endswith(')'):
        inner_s = s[1:-1]
        inner_type_str_list = [repair_type_str(s1) for s1 in inner_s.split(',')]
        return f'tuple[{", ".join(inner_type_str_list)}]'
    if s.endswith('tuple of numbers / None'):
        return 'tuple[int] | None'

    parts = s.replace(',', ' ').replace(' and ', ' ').replace(' or ', ' ').replace("'", '').split()
    return build_container_type_str(parts)

def three_quote_line(s: str):
    if '"""' not in s:
        return '"""' + s + '"""\n'
    elif "'''" not in s:
        return "'''" + s + "'''\n"
    else:
        return '"""' + s.replace('"""', '') + '"""\n'

class Member:
    def __init__(self):
        self.declaration = ''
        self.doc_string = ''
        self.is_static = False
        self.is_property = False

    def dump_pyi(self, indent: str, in_class: bool, type_name_resolve: Callable[[str], str]):
        def replace_py_type(m: re.Match):
            # the matched string is a type string, may include :: wrongly, replace it with .
            s = m.group(0).replace('::', '.')
            return type_name_resolve(s)
        
        declaration = self.declaration

        declaration = typo_fix(declaration)

        parts = declaration.split('`')
        for i in range(0, len(parts)):
            # Inside and outside the `` marks, use different regex to remove [] link syntax.
            # Inside the `` marks, replace links like [Hom:hou.parmCondType] but do not replace links like [hou.parmCondType].
            r = link_re if i % 2 == 0 else link_in_quote_re
            parts[i] = r.sub(lambda m: m.group('name'), parts[i])

        # delete all the ``
        declaration = ''.join(parts)

        # delete unused =Hom:
        declaration = declaration.replace('=Hom:', '=')

        # replace all py type
        declaration = py_type_re.sub(replace_py_type, declaration)

        doc_string = three_quote_line(self.doc_string) if self.doc_string else ''

        if self.is_property:
            if in_class and not self.is_static:
                # convert property to property method
                if ':' in declaration:
                    name, type_str = (s.strip() for s in declaration.split(':', 1))
                    type_str = repair_type_str(type_str)
                    declaration = f'def {name}(self) -> {type_str}:\n'
                else:
                    name = declaration
                    declaration = f'def {name}(self):\n'
                
                return indent + '@property\n' + indent + declaration + indent + '    ' + (doc_string or '...') + '\n'
            else:
                return indent + declaration + ' = ...\n' + indent + doc_string + '\n'
                
        else:
            # If it's a static method, add @staticmethod
            decoration_line = indent + '@staticmethod\n' if in_class and self.is_static else ''
            if '->' in declaration:
                parameter_str, return_type_str = (s.strip() for s in declaration.split('->', 1))
                if '(' in parameter_str and ')' in parameter_str:
                    # fix typo
                    parameter_str = parameter_str.replace(', bool ', ', ').replace('):, ', ', ')
                    # fix optional param like: onWorkItemSetStringAttrib(self, workitem_id, index, attribute_name, value, [attrib_index])
                    parameter_str = optinal_param_re.sub(repair_optional_param_str, parameter_str)
                elif self.is_static:
                    parameter_str += '()'
                else:
                    parameter_str += '(self)'
                return_type_str = repair_type_str(return_type_str)
                declaration = f'{parameter_str} -> {return_type_str}'
            else:
                declaration = declaration.replace(', bool ', ', ').replace('):, ', ', ')
                declaration = optinal_param_re.sub(repair_optional_param_str, declaration)

            return decoration_line + indent + f'def {declaration}:\n' + indent + '    ' + (doc_string or '...') + '\n'

@dataclass
class MetaData:
    path: tuple[str, ...]
    meta_type: MetaType
    summary: str
    doc_string: str
    super_class: str
    member_list: list[Member]

class Node:
    name: str
    children: dict[str, 'Node']
    meta_data: Optional[MetaData]

    def __init__(self, name: str):
        self.name = name
        self.children = {}
        self.meta_data = None

    def insert(self, meta_data: MetaData):
        path = meta_data.path
        node = self
        for name in path:
            if name not in node.children:
                node.children[name] = Node(name)
            node = node.children[name]
        node.meta_data = meta_data

    def print(self, indent: int = 0):
        summary = self.meta_data.summary if self.meta_data is not None else ''
        meta_type = self.meta_data.meta_type if self.meta_data is not None else ''
        print(f"{' ' * indent}{meta_type} {self.name} {summary}")
        for child in self.children.values():
            child.print(indent + 2)

    def export_pyi(self, root_path: str):
        assert self.meta_data is None or self.meta_data.meta_type in (MetaType.Package, MetaType.Module)
        stub_lines: list[str] = []

        # write module-level comment
        if self.meta_data is not None:
            doc_string = three_quote_line(self.meta_data.summary + '\n' + self.meta_data.doc_string)
            stub_lines.append(doc_string)

        # write import
        stub_lines.append('from enum import Enum\nfrom typing import Generator, Iterable, Any, Union, Sequence, Collection, Callable\n')

        import_line_index = len(stub_lines)

        stub_lines.append('\n')

        package_need_import: set[str] = set()

        def type_name_resolve(name: str) -> str:
            if '.' not in name:
                raise NotImplementedError

            package_name, type_name = name.split('.', 1)

            if package_name.startswith('_'):
                # Some places mistakenly used package names starting with _
                package_name = package_name[1:]

            if package_name == self.name:
                # When referencing types from the current package, return the type name directly.
                return type_name
            else:
                node = self.children.get(package_name)
                if node is not None and node.meta_data is not None and node.meta_data.meta_type in (MetaType.Class, MetaType.Enum):
                    # The referenced items are in this file.
                    return name

                # When referencing types from other packages, an import is needed.
                package_need_import.add(package_name)
                return name

        # Write module-level member
        if self.meta_data is not None:
            for member in self.meta_data.member_list:
                stub_lines.append(member.dump_pyi('', False, type_name_resolve))

        any_inner_package = False
        any_member = False

        # Write class, enum, function
        for name, child in self.children.items():
            if child.meta_data is None or child.meta_data.meta_type in (MetaType.Package, MetaType.Module):
                child.export_pyi(os.path.join(root_path, name))
                any_inner_package = True
            elif child.meta_data.meta_type == MetaType.Class:
                stub_lines.append(child.dump_class_pyi('', type_name_resolve))
                any_member = True
            elif child.meta_data.meta_type == MetaType.Enum:
                stub_lines.append(child.dump_enum_pyi(''))
                any_member = True
            elif child.meta_data.meta_type == MetaType.Function:
                if len(child.meta_data.member_list) > 0:
                    stub_lines.append(child.meta_data.member_list[0].dump_pyi('', False, type_name_resolve))
                    any_member = True
            else:
                raise NotImplementedError
            
        # Insert import
        import_lines = [f'import {package_name}\n' for package_name in package_need_import]
        stub_lines[import_line_index:import_line_index] = import_lines

        if self.meta_data is not None or any_member:
            # Write pyi file
            if any_inner_package:
                # If there are internal packages, they need to be declared in the current package's __init__.pyi file.
                os.makedirs(root_path, exist_ok=True)
                output_path = os.path.join(root_path, '__init__.pyi')
            else:
                output_path = root_path + '.pyi'
                dir_path = os.path.dirname(output_path)
                os.makedirs(dir_path, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(''.join(stub_lines))

    def dump_class_pyi(self, indent: str, type_name_resolve: Callable[[str], str]) -> str:
        assert self.meta_data is not None and self.meta_data.meta_type == MetaType.Class

        stub_lines: list[str] = []

        # Write class declaration
        super_class_str = f'({type_name_resolve(self.meta_data.super_class)})' if self.meta_data.super_class else ''
        class_declaration = indent + f'class {self.name}{super_class_str}:\n'
        stub_lines.append(class_declaration)

        indent_inner = indent + '    '
        # Write doc string
        doc_string = indent_inner + three_quote_line(self.meta_data.summary + '\n' + self.meta_data.doc_string)
        stub_lines.append(doc_string)

        # Write members
        for member in self.meta_data.member_list:
            stub_lines.append(member.dump_pyi(indent_inner, True, type_name_resolve))
        
        # Write nested classes and enums
        for child in self.children.values():
            assert child.meta_data is not None
            if child.meta_data.meta_type == MetaType.Class:
                stub_lines.append(child.dump_class_pyi(indent_inner, type_name_resolve))
            elif child.meta_data.meta_type == MetaType.Enum:
                stub_lines.append(child.dump_enum_pyi(indent_inner))
            else:
                raise NotImplementedError
            
        return ''.join(stub_lines)

    def dump_enum_pyi(self, indent: str) -> str:
        assert self.meta_data is not None and self.meta_data.meta_type == MetaType.Enum

        stub_lines: list[str] = []

        # Write enum declarations
        enum_declaration = indent + f'class {self.name}(Enum):\n'
        stub_lines.append(enum_declaration)

        indent_inner = indent + '    '
        # Write doc string
        doc_string = indent_inner + three_quote_line(self.meta_data.summary + '\n' + self.meta_data.doc_string)
        stub_lines.append(doc_string)

        full_name_prefix = '.'.join(self.meta_data.path) + '.'

        def type_name_resolve(name: str) -> str:
            if name.startswith(full_name_prefix):
                return name[len(full_name_prefix):]
            else:
                return name

        # Write enum members
        for member in self.meta_data.member_list:
            stub_lines.append(member.dump_pyi(indent_inner, True, type_name_resolve))

        assert len(self.children) == 0

        return ''.join(stub_lines)

def return_function_meta_data(path: tuple[str, ...], declaration_line: str, doc_string: str) -> MetaData:
    member = Member()
    if '.'.join(path) == 'hou.startHoudiniEngineDebugger':
        member.declaration = 'startHoudiniEngineDebugger(arg: int | str) -> None'
    else:
        member.declaration = declaration_line[len(':usage:'):].strip(': \n.')

    member.doc_string = doc_string
    member.is_static = True
    return MetaData(
        path=path,
        meta_type=MetaType.Function,
        summary='',
        doc_string='',
        super_class='',
        member_list=[member])


def parse(reader: io.TextIOBase) -> MetaData | None:
    title_line: str = next(reader)
    if title_line.startswith('\ufeff'):
        title_line = title_line[1:]

    # The content of title_line is similar to `= pdg.acceptResult =`, which needs to be validated and then `pdg.acceptResult` should be extracted.
    if not title_line.startswith('=') or not title_line.endswith('=\n'):
        return None
    title = title_line[1:-2].strip()
    path = tuple(title.split('.'))

    summary = ''
    meta_type = ''
    super_class = ''
    declaration_line = ''
    main_doc_lines: list[str] = []

    for line in reader:
        if line.startswith('"""'):
            # This line is the summary
            if line.endswith('"""\n'):
                summary = line[3:-4]
                break
            summary_lines = [line[3:]]
            for line in reader:
                if line.endswith('"""\n'):
                    summary_lines.append(line[:-4])
                    summary = ''.join(summary_lines)
                    break
                summary_lines.append(line)
            else:
                # Did not find the ending """, indicating an issue with this file.
                return None
            break
        if line.startswith(':usage:'):
            # No summary, but there is usage, indicating this is a function.
            if name_to_meta_type.get(meta_type) != MetaType.Function:
                # Might have encountered a vex function, not a Python function
                return None
            return return_function_meta_data(path, line, ''.join(reader))
        if line.startswith('#') and line[1] not in (' ', '\t', '!') and not line.startswith('#pragma '):
            # This is a meta line, such as `#type: pypackage`
            key, value = line[1:].split(':', 1)
            if key == 'type':
                meta_type = value.strip()
            elif key == 'superclass':
                super_class = value.strip()
        elif line.startswith('== '):
            # Before encountering a summary, there should not be a second title.
            return None

    meta_type = name_to_meta_type.get(meta_type, None)
    if meta_type is None:
        return None

    is_subtopics = False

    parse_state = 0

    # The part before the @ symbol is the doc string
    for line in reader:
        if line.startswith('@'):
            if line.startswith('@subtopics'):
                is_subtopics = True
            break
        elif line.startswith('::'):
            # This line declares a member, skipping lines that start with @
            declaration_line = line
            parse_state = 1
            break
        elif line.startswith(':usage:'):
            if meta_type != MetaType.Function:
                # Might have encountered a vex function, not a Python function
                return None
            # Indicating this is a function.
            return return_function_meta_data(path, line, summary + ''.join(main_doc_lines) + ''.join(reader))
        main_doc_lines.append(line)

    # Remove leading and trailing blank lines from main_doc_lines
    while main_doc_lines and main_doc_lines[0].strip() == '':
        main_doc_lines.pop(0)
    while main_doc_lines and main_doc_lines[-1].strip() == '':
        main_doc_lines.pop()

    main_doc = ''.join(main_doc_lines)

    member_list: list[Member] = []

    arrow_typo_list = [' : ', ' -:> ',  ' - > ', ' - ', ': -> ']

    if not is_subtopics:

        current_member = Member()
        current_doc_string_list: list[str] = []

        def parse_declaration(line: str):
            declaration = line.strip(': \n')

            has_bracket = '(' in declaration and ')' in declaration

            if has_bracket:
                # It's a function, but -> is written incorrectly, it needs to be replaced back
                for arrow_typo in arrow_typo_list:
                    if arrow_typo in declaration:
                        declaration = declaration.replace(arrow_typo, ' -> ')
                        break

            current_member.is_property = not has_bracket and not '->' in declaration
            current_member.declaration = declaration

            # If it's not a class, assume it's static
            current_member.is_static = meta_type != MetaType.Class

        def end_member():
            nonlocal current_member
            nonlocal current_doc_string_list
            nonlocal member_list
            nonlocal parse_state

            # Remove leading and trailing blank lines from current_doc_string_list
            while current_doc_string_list and current_doc_string_list[0].strip() == '':
                current_doc_string_list.pop(0)
            while current_doc_string_list and current_doc_string_list[-1].strip() == '':
                current_doc_string_list.pop()

            current_member.doc_string = ''.join(current_doc_string_list)
            current_doc_string_list = []
            member_list.append(current_member)
            current_member = Member()
            parse_state = 0

       

        for line in reader:
            if parse_state == 0:
                # Expecting member declarations to start with ::
                if line.startswith('::'):
                    declaration_line = line

                    parse_state = 1
            elif parse_state > 0:
                if parse_state == 1:
                    # Expecting the first few lines of a member's doc string to contain meta information, starting with #
                    still_declaration = False
                    if declaration_line:
                        if line[0] != ' ' and line[0] != '\t' and not line.startswith('::'):
                            # Indicates that the declaration has wrapped
                            declaration_line = declaration_line[:-1] + line[:-1]
                            still_declaration = True
                        parse_declaration(declaration_line)
                        declaration_line = ''
                        if still_declaration:
                            # This line is still part of the declaration, process the next line
                            continue
                    if line.startswith('    #') or line.startswith('\t#'):
                        if line.startswith('    #mode: static') or line.startswith('\t#mode: static'):
                            current_member.is_static = True
                    else:
                        parse_state = 2
                if line.startswith('    ') or line.startswith('\t') or line == '\n':
                    current_doc_string_list.append(line)
                else:
                    end_member()
                    if line.startswith('::'):
                        declaration_line = line
                        parse_state = 1
                    else:
                        parse_state = 0
        
        if parse_state > 0:
            if declaration_line:
                parse_declaration(declaration_line)
                declaration_line = ''
            end_member()

    if meta_type == MetaType.Module:
        # If it's a module and all its members are of enum type, then it is considered an enum
        for member in member_list:
            if ':' in member.declaration or '>' in member.declaration or '-' in member.declaration:
                break
        else:
            meta_type = MetaType.Enum

    return MetaData(
        path=path,
        meta_type=meta_type,
        summary=summary,
        doc_string=main_doc,
        super_class=super_class,
        member_list=member_list)

def run_text(text_input_root_dir: str, output_root_dir = 'output', dry_run = False):
    node_tree = Node('')

    for root, dirs, files in os.walk(text_input_root_dir):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    result = parse(f)

                if result is not None:
                    node_tree.insert(result)
    if dry_run:
        node_tree.print()
    else:
        node_tree.export_pyi(output_root_dir)

def run_zip(houdini_root_dir: str, output_root_dir = 'output', dry_run = False):
    input_dir = os.path.join(houdini_root_dir, 'houdini', 'help')
    input_file_list = ['hom.zip', 'tops.zip']
    node_tree = Node('')

    # Find all zip files in input_dir
    for file in input_file_list:
        input_path = os.path.join(input_dir, file)
        # Read all txt files from the zip file
        import zipfile
        with zipfile.ZipFile(input_path, 'r') as z:
            for file in z.namelist():
                if file.endswith('.txt'):
                    with z.open(file, 'r') as f:
                        text_stream = io.TextIOWrapper(f, encoding='utf-8')
                        result = parse(text_stream)

                    if result is not None:
                        node_tree.insert(result)

    if dry_run:
        node_tree.print()
    else:
        node_tree.export_pyi(output_root_dir)

def main():
    parser = argparse.ArgumentParser(description='Parse Houdini help text files to generate pyi files.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--houdini-input', type=str, help='Houdini installation directory as input.')
    group.add_argument('--text-input', type=str, help='Input directory containing help text files to parse.')
    parser.add_argument('--output', type=str, help='Output directory to store pyi files.', default='output')
    parser.add_argument('--dry-run', action='store_true', help='Print info without generating pyi files.')
    args = parser.parse_args()

    if args.houdini_input:
        run_zip(args.houdini_input, args.output, args.dry_run)
    elif args.text_input:
        run_text(args.text_input, args.output, args.dry_run)

if __name__ == '__main__':
    main()
