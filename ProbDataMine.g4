grammar ProbDataMine;

// Parser Rules
program : statement+ EOF ;

statement 
    : dataLoadStmt 
    | variableDeclaration 
    | variableAssignment
    | queryStmt 
    | clusteringStmt 
    | associationStmt 
    | classificationStmt 
    | commentStmt
    ;

dataLoadStmt : LOAD_DATA '(' STRING (',' loadOption)* ')' SEMICOLON ;
loadOption : IDENTIFIER ':' expr ;

variableDeclaration : VAR IDENTIFIER FOLLOWS distributionExpr SEMICOLON ;
variableAssignment : VAR IDENTIFIER '=' expr SEMICOLON ;

distributionExpr 
    : NORMAL '(' expr ',' expr ')'
    | LOGNORMAL '(' expr ',' expr ')'
    | POISSON '(' expr ')'
    | BERNOULLI '(' expr ')'
    | EMPIRICAL_DISTRIBUTION '(' dataRef ')'
    | GAMMA '(' expr ',' expr ')'
    | BETA '(' expr ',' expr ')'
    | MULTINOMIAL '(' expr (',' expr)* ')'
    | FITTED_TO ':' dataRef
    ;

dataRef : DATA '.' IDENTIFIER ;

queryStmt : QUERY queryExpr SEMICOLON ;
queryExpr 
    : 'P' '(' conditionalExpr ')'
    | 'E' '(' expr ('|' conditionalExpr)? ')'
    | 'correlation' '(' expr ',' expr ')'
    | 'outliers' '(' expr (',' expr)* ')'
    ;

clusteringStmt : CLUSTER '(' IDENTIFIER ',' clusteringOptions ')' SEMICOLON ;
clusteringOptions : 'dimensions' ':' '[' expr (',' expr)* ']' ',' 'k' ':' INTEGER ;

associationStmt : FIND_ASSOCIATIONS '(' IDENTIFIER (',' associationOption)* ')' SEMICOLON ;
associationOption : 'min_support' ':' expr | 'min_confidence' ':' expr ;

classificationStmt : CLASSIFY '(' IDENTIFIER ',' 'target' ':' expr (',' classifierOption)* ')' SEMICOLON ;
classifierOption : IDENTIFIER ':' expr ;

expr : conditionalExpr ;
conditionalExpr : logicalOrExpr ('?' logicalOrExpr ':' logicalOrExpr)? ;
logicalOrExpr : logicalAndExpr (OR logicalAndExpr)* ;
logicalAndExpr : comparisonExpr (AND comparisonExpr)* ;
comparisonExpr : addExpr (comparisonOp addExpr)? ;
addExpr : multExpr (('+' | '-') multExpr)* ;
multExpr : powExpr (('*' | '/') powExpr)* ;
powExpr : unaryExpr ('^' unaryExpr)* ;
unaryExpr : '-' unaryExpr | primary ;
primary : INTEGER | FLOAT | IDENTIFIER ('.' IDENTIFIER)* | '(' expr ')' ;
comparisonOp : '>' | '<' | '>=' | '<=' | '==' | '!=' ;

commentStmt : COMMENT ;

// Lexer Rules
COMMENT : '//' ~[\r\n]* ;
STRING : '"' (~["\r\n])* '"' ;
IDENTIFIER : [a-zA-Z][a-zA-Z0-9_]* ;
INTEGER : [0-9]+ ;
FLOAT : [0-9]+ '.' [0-9]* | '.' [0-9]+ ;

// Keywords
LOAD_DATA : 'load_data' ;
VAR : 'var' ;
FOLLOWS : 'follows' ;
QUERY : 'query' ;
CLUSTER : 'cluster' ;
FIND_ASSOCIATIONS : 'find_associations' ;
CLASSIFY : 'classify' ;
DATA : 'data' ;

// Distribution types
NORMAL : 'Normal' ;
LOGNORMAL : 'LogNormal' ;
POISSON : 'Poisson' ;
BERNOULLI : 'Bernoulli' ;
EMPIRICAL_DISTRIBUTION : 'EmpiricalDistribution' ;
GAMMA : 'Gamma' ;
BETA : 'Beta' ;
MULTINOMIAL : 'Multinomial' ;
FITTED_TO : 'fitted_to' ;

// Logical operators
AND : 'and' ;
OR : 'or' ;
NOT : 'not' ;

// Punctuation
SEMICOLON : ';' ;

WS : [ \t\r\n]+ -> skip ;