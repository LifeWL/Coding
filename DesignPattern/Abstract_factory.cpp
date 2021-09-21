#include <iostream>
#include <vector>
using namespace std;

// Here is the product class
class Operation_Pos
{
public:
    int var1, var2;
    virtual double GetResult()
    {
        double res = 0;
        return res;
    }
};

class Add_Operation_Pos : public Operation_Pos
{
public:
    virtual double GetResult()
    {
        return var1 + var2;
    }
};

class Sub_Operation_Pos : public Operation_Pos
{
public:
    virtual double GetResult()
    {
        return var1 - var2;
    }
};

class Mul_Operation_Pos : public Operation_Pos
{
public:
    virtual double GetResult()
    {
        return var1 * var2;
    }
};

class Div_Operation_Pos : public Operation_Pos
{
public:
    virtual double GetResult()
    {
        return var1 / var2;
    }
};
/*********************************************************************************/
class Operation_Neg
{
public:
    int var1, var2;
    virtual double GetResult()
    {
        double res = 0;
        return res;
    }
};

class Add_Operation_Neg : public Operation_Neg
{
public:
    virtual double GetResult()
    {
        return -(var1 + var2);
    }
};

class Sub_Operation_Neg : public Operation_Neg
{
public:
    virtual double GetResult()
    {
        return -(var1 - var2);
    }
};

class Mul_Operation_Neg : public Operation_Neg
{
public:
    virtual double GetResult()
    {
        return -(var1 * var2);
    }
};

class Div_Operation_Neg : public Operation_Neg
{
public:
    virtual double GetResult()
    {
        return -(var1 / var2);
    }
};
/*****************************************************************************************************/

// Here is the Factory class
class Factory
{
public:
    virtual Operation_Pos *CreateProduct_Pos() = 0;
    virtual Operation_Neg *CreateProduct_Neg() = 0;
};

class Add_Factory : public Factory
{
public:
    Operation_Pos *CreateProduct_Pos()
    {
        return new Add_Operation_Pos();
    }
    Operation_Neg *CreateProduct_Neg()
    {
        return new Add_Operation_Neg();
    }
};

class Sub_Factory : public Factory
{
public:
    Operation_Pos *CreateProduct_Pos()
    {
        return new Sub_Operation_Pos();
    }
    Operation_Neg *CreateProduct_Neg()
    {
        return new Sub_Operation_Neg();
    }
};

class Mul_Factory : public Factory
{
public:
    Operation_Pos *CreateProduct_Pos()
    {
        return new Mul_Operation_Pos();
    }
    Operation_Neg *CreateProduct_Neg()
    {
        return new Mul_Operation_Neg();
    }
};

class Div_Factory : public Factory
{
public:
    Operation_Pos *CreateProduct_Pos()
    {
        return new Div_Operation_Pos();
    }
    Operation_Neg *CreateProduct_Neg()
    {
        return new Div_Operation_Neg();
    }
};

int main()
{
    int a, b;
    cin >> a >> b;
    Add_Factory *p_fac = new Add_Factory();
    Operation_Pos *p_pro = p_fac->CreateProduct_Pos();
    p_pro->var1 = a;
    p_pro->var2 = b;
    cout << p_pro->GetResult() << endl;

    Add_Factory *p_fac1 = new Add_Factory();
    Operation_Neg *p_pro1 = p_fac1->CreateProduct_Neg();
    p_pro1->var1 = a;
    p_pro1->var2 = b;
    cout << p_pro1->GetResult() << endl;

    return 0;
}
