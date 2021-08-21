#  字符串的调整II

```C++
#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

void replace_space(string &str) {
    int space_count = 0;
    for (int i = 0; i < str.size(); ++i) {
        if (str[i] == ' '){
            space_count ++;
        }
    }
    int pos1 = str.size() - 1;
    str.resize(str.size() + 2* space_count);
    int pos2 = str.size() - 1;
    while (pos1 >= 0) {
        if (str[pos1] != ' ') {
            str[pos2--] = str[pos1--];
        }else {
            str[pos2--] = '0';
            str[pos2--] = '2';
            str[pos2--] = '%';
            pos1 --;
        }
    }
    cout << str;
}

int main() {
    string str;
    getline(cin, str);
    replace_space(str);
}

```

