from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

def printlist(lst, f=str, start="[", sepa=",", end="]"):
    return start + sepa.join(f(i) for i in lst) + end

class AST(ABC):
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @abstractmethod
    def accept(self, v, param):
        return v.visit(self, param)

@dataclass
class Program(AST):
    statements: List['Statement']
    
    def __str__(self):
        return f"Program({printlist(self.statements)})"
    
    def accept(self, v, param):
        return v.visitProgram(self, param)

@dataclass
class Statement(AST):
    __metaclass__ = ABCMeta
    pass

@dataclass
class DataLoadStmt(Statement):
    filename: str
    options: Dict[str, str]
    
    def __str__(self):
        return f"LoadData({self.filename}, {self.options})"
    
    def accept(self, v, param):
        return v.visitDataLoadStmt(self, param)

@dataclass
class VariableDecl(Statement):
    name: str
    distribution: 'Distribution'
    
    def __str__(self):
        return f"VarDecl({self.name}, {self.distribution})"
    
    def accept(self, v, param):
        return v.visitVariableDecl(self, param)

@dataclass
class Distribution(AST):
    type: str
    params: List['Expression']
    
    def __str__(self):
        return f"{self.type}({printlist(self.params)})"
    
    def accept(self, v, param):
        return v.visitDistribution(self, param)

@dataclass
class Query(Statement):
    type: str  # P, E, correlation, outliers
    params: List['Expression']
    condition: Optional['Expression'] = None
    
    def __str__(self):
        base = f"Query{self.type}({printlist(self.params)})"
        if self.condition:
            base += f" | {self.condition}"
        return base
    
    def accept(self, v, param):
        return v.visitQuery(self, param)

@dataclass
class Expression(AST):
    __metaclass__ = ABCMeta
    pass

@dataclass
class BinaryOp(Expression):
    op: str
    left: Expression
    right: Expression
    
    def __str__(self):
        return f"BinOp({self.op}, {self.left}, {self.right})"
    
    def accept(self, v, param):
        return v.visitBinaryOp(self, param)

@dataclass
class Literal(Expression):
    value: Any
    
    def __str__(self):
        return f"Literal({self.value})"
    
    def accept(self, v, param):
        return v.visitLiteral(self, param)

@dataclass
class Variable(Expression):
    name: str
    
    def __str__(self):
        return f"Var({self.name})"
    
    def accept(self, v, param):
        return v.visitVariable(self, param)