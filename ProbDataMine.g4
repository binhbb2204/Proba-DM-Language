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
LOAD_DATA : 'load_data' ;
loadOption : IDENTIFIER ':' expr ;

variableDeclaration : VAR IDENTIFIER FOLLOWS distributionExpr SEMICOLON ;
FOLLOWS : 'follows' ;
variableAssignment : VAR IDENTIFIER '=' expr SEMICOLON ;
VAR : 'var' ;
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
NORMAL : 'Normal' ;
LOGNORMAL : 'LogNormal' ;
POISSON : 'Poisson' ;
BERNOULLI : 'Bernoulli' ;
EMPIRICAL_DISTRIBUTION : 'EmpiricalDistribution' ;
GAMMA : 'Gamma' ;
BETA : 'Beta' ;
MULTINOMIAL : 'Multinomial' ;
FITTED_TO : 'fitted_to' ;
dataRef : DATA '.' IDENTIFIER ;
DATA : 'data' ;

queryStmt : QUERY queryExpr SEMICOLON ;
QUERY : 'query' ;
queryExpr 
    : P '(' expr (OR_OP expr)? ')'            // Allow conditional P(A | B)
    | E '(' expr (OR_OP expr)? ')'            // Allow conditional E(X | Y)
    | CORRELATION '(' expr ',' expr ')'
    | OUTLIERS '(' expr (',' expr)* ')'
    ;
P: 'P';
E: 'E';
CORRELATION: 'correlation';
OUTLIERS: 'outliers';
OR_OP : '|' ;

clusteringStmt : CLUSTER '(' IDENTIFIER ',' clusteringOptions ')' SEMICOLON ;
CLUSTER : 'cluster' ;
clusteringOptions : 'dimensions' ':' '[' expr (',' expr)* ']' ',' 'k' ':' INTEGER ;

associationStmt : FIND_ASSOCIATIONS '(' IDENTIFIER (',' associationOption)* ')' SEMICOLON ;
FIND_ASSOCIATIONS : 'find_associations' ;
associationOption : 'min_support' ':' expr | 'min_confidence' ':' expr ;

classificationStmt : CLASSIFY '(' IDENTIFIER ',' 'target' ':' expr (',' classifierOption)* ')' SEMICOLON ;
CLASSIFY : 'classify' ;
classifierOption : IDENTIFIER ':' expr ;

expr : conditionalExpr ;
conditionalExpr : logicalOrExpr ('?' logicalOrExpr ':' logicalOrExpr)? ;
logicalOrExpr : logicalAndExpr (OR logicalAndExpr)* ;
OR : 'or' ;
logicalAndExpr : comparisonExpr (AND comparisonExpr)* ;
AND : 'and' ;
comparisonExpr : addExpr (comparisonOp addExpr)? ;
comparisonOp : '>' | '<' | '>=' | '<=' | '==' | '!=' ;

addExpr : multExpr (('+' | '-') multExpr)* ;
multExpr : powExpr (('*' | '/') powExpr)* ;
powExpr : unaryExpr ('^' unaryExpr)* ;
unaryExpr : '-' unaryExpr | primary ;
primary : INTEGER | FLOAT | IDENTIFIER ('.' IDENTIFIER)* | '(' expr ')' ;

commentStmt : COMMENT ;

// Lexer Rules
COMMENT : '//' ~[\n]* '\n'? -> skip ;
STRING : '"' (~["\r\n])* '"' ;
IDENTIFIER : [a-zA-Z][a-zA-Z0-9_]* ;
INTEGER : [0-9]+ ;
FLOAT : [0-9]+ '.' [0-9]* | '.' [0-9]+ ;

SEMICOLON : ';' ;
LPAREN : '(' ;
RPAREN : ')' ;
COMMA : ',' ;
COLON : ':' ;
LBRACK : '[' ;
RBRACK : ']' ;

WS : [ \t\r\n]+ -> skip ;
