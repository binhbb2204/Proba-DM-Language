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

dataLoadStmt : LOAD_DATA LPAREN STRING (COMMA loadOption)* RPAREN SEMICOLON ;
LOAD_DATA : 'load_data' ;
loadOption : IDENTIFIER COLON expr ;

variableDeclaration : VAR IDENTIFIER FOLLOWS distributionExpr SEMICOLON ;
FOLLOWS : 'follows' ;
variableAssignment : VAR IDENTIFIER '=' expr SEMICOLON ;
VAR : 'var' ;
distributionExpr 
    : NORMAL LPAREN expr COMMA expr RPAREN
    | LOGNORMAL LPAREN expr COMMA expr RPAREN
    | POISSON LPAREN expr RPAREN
    | BERNOULLI LPAREN expr RPAREN
    | EMPIRICAL_DISTRIBUTION LPAREN dataRef RPAREN
    | GAMMA LPAREN expr COMMA expr RPAREN
    | BETA LPAREN expr COMMA expr RPAREN
    | MULTINOMIAL LPAREN expr (COMMA expr)* RPAREN
    | FITTED_TO COLON dataRef
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
    : P LPAREN expr (OR_OP expr)? RPAREN            // Allow conditional P(A | B)
    | E LPAREN expr (OR_OP expr)? RPAREN            // Allow conditional E(X | Y)
    | CORRELATION LPAREN expr COMMA expr RPAREN
    | OUTLIERS LPAREN expr (COMMA expr)* RPAREN
    ;
P: 'P';
E: 'E';
CORRELATION: 'correlation';
OUTLIERS: 'outliers';
OR_OP : '|' ;

clusteringStmt : CLUSTER LPAREN IDENTIFIER COMMA clusteringOptions RPAREN SEMICOLON ;
CLUSTER : 'cluster' ;
clusteringOptions : 'dimensions' COLON LBRACK expr (COMMA expr)* RBRACK COMMA 'k' COLON INTEGER ;

associationStmt : FIND_ASSOCIATIONS LPAREN IDENTIFIER (COMMA associationOption)* RPAREN SEMICOLON ;
FIND_ASSOCIATIONS : 'find_associations' ;
associationOption : 'min_support' COLON expr | 'min_confidence' COLON expr ;

classificationStmt : CLASSIFY LPAREN IDENTIFIER COMMA 'target' COLON expr (COMMA classifierOption)* RPAREN SEMICOLON ;
CLASSIFY : 'classify' ;
classifierOption : IDENTIFIER COLON expr ;

expr : conditionalExpr ;
conditionalExpr : logicalOrExpr ('?' logicalOrExpr COLON logicalOrExpr)? ;
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
primary : INTEGER | FLOAT | IDENTIFIER ('.' IDENTIFIER)* | LPAREN expr RPAREN ;

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
