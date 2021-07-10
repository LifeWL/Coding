
class Solution {
public:
    string replaceSpace(string s) {
       int count = 0, len = s.size();
       for(char c : s) {
           if (c == ' ') count++;
       } 
       s.resize(count * 2 + len); 
       for (int i = s.size() - 1, j = len - 1; j < i; i--, j--) {
           if (s[j] != ' ') s[i] = s[j];
           else {
               s[i - 2] = '%';
               s[i - 1] = '2';
               s[i] = '0';
               i -= 2; 
               }
       }
        return s;
    }
};


