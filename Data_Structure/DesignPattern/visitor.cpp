#include <iostream>
using namespace std;

struct B;
struct C;
struct D;
struct E;
struct F;

struct A {
    struct IVisitor {
        virtual void visit(B *) = 0;
        virtual void visit(C *) = 0;
        virtual void visit(D *) = 0;
        virtual void visit(E *) = 0;
        virtual void visit(F *) = 0;
    };
    virtual void Accept(IVisitor *) = 0;
    virtual ~A() {}
};

struct B : A {
    B() {
        x = 12567;
    }
    virtual void Accept(IVisitor *vis) {
        vis->visit(this);
    }
    int x;
};

struct C : A {
    C() {
        x = 90.87;
    }
    virtual void Accept(IVisitor *vis) {
        vis->visit(this);
    }
    double x;
};

struct D : A {
    D() {
        x = "hello world";
    }
    virtual void Accept(IVisitor *vis) {
        vis->visit(this);
    }
    string x;
};

struct E : A {
    E() {
        x = new int;
    }
    virtual void Accept(IVisitor *vis) {
        vis->visit(this);
    }
    int *x;
};

struct F : A {
    F() {
        x = new int;
    }
    virtual void Accept(IVisitor *vis) {
        vis->visit(this);
    }
    int *x;
};

struct OutputVisitor : A::IVisitor {
    virtual void visit(B *obj) {
        cout << "Class B" << endl;
    }
    virtual void visit(C *obj) {
        cout << "Class C" << endl;
    }
    virtual void visit(D *obj) {
        cout << "Class D" << endl;
    }
    virtual void visit(E *obj) {
        cout << "Class E" << endl;
    }
    virtual void visit(F *obj) {
        cout << "Class F" << endl;
    }
};

struct CalcVisitor : A::IVisitor {
    CalcVisitor(int val) : val(val) {}
    virtual void visit(B *obj) {
        val += 2;
    }
    virtual void visit(C *obj) {
        val *= 3;
    }
    virtual void visit(D *obj) {
        val -= 5;
    }
    virtual void visit(E *obj) {
        val += 6;
    }
    virtual void visit(F *obj) {
        val *= 2;
    }
    int val;
};

ostream &operator<<(ostream &out, const CalcVisitor &vis) {
    out << vis.val;
    return out;
}

int main() {
    A *pa_arr[10];
    for (int i = 0; i < 10; i++) {
        switch (rand() % 5) {
            case 0: pa_arr[i] = new B(); break;
            case 1: pa_arr[i] = new C(); break;
            case 2: pa_arr[i] = new D(); break;
            case 3: pa_arr[i] = new E(); break;
            case 4: pa_arr[i] = new F(); break;
        }
    }
    CalcVisitor ans(1);
    OutputVisitor vis;
    for (int i = 0; i < 10; i++) {
        pa_arr[i]->Accept(&vis);
        pa_arr[i]->Accept(&ans);
    }
    cout << ans << endl;
    return 0;
}
