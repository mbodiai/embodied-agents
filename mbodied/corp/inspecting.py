import inspect
from typing import Any, Dict
def collect_members(obj: Any, depth: int = 1, current_depth: int = 0, include_signatures: bool = False, include_docstrings: bool = True) -> Dict[str, Any]:
    if current_depth > depth:
        return {}
    
    members_dict = {}
    members = dir(obj)
    
    for member in members:
        if member.startswith("__") and member.endswith("__"):
            continue
        
        member_obj = getattr(obj, member)
        member_info = {}
        
        if inspect.isclass(member_obj) or inspect.ismodule(member_obj):
            member_info["type"] = "class" if inspect.isclass(member_obj) else "module"
            if include_docstrings:
                member_info["docstring"] = inspect.getdoc(member_obj)
            member_info["members"] = collect_members(member_obj, depth, current_depth + 1, include_signatures, include_docstrings)
        else:
            member_info["type"] = "function" if inspect.isfunction(member_obj) else "attribute"
            if include_docstrings:
                member_info["docstring"] = inspect.getdoc(member_obj)
            if include_signatures and inspect.isfunction(member_obj):
                member_info["signature"] = str(inspect.signature(member_obj))
        
        members_dict[member] = member_info
    
    return members_dict

# Example usage
if __name__ == "__main__":
    from mbodied.agents import Agent 
    from rich import print
    print(collect_members(Agent, depth=2, include_signatures=True))