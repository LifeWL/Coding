```c
int strcmp(const char* p1, const char* p2)
{
  const unsigned char* s1 = (const unsigned char*)p1;
  const unsigned char* s2 = (const unsigned char*)p2;
  unsigned char c1, c2;
  do {
    c1 = (unsigned char) *s1++;
    c2 = (unsigned char) *s2++;
    if (c1 == '\0') {
      return c1 - c2;
    }
  } while(c1 == c2);
  return c1 - c2;
}
```


```cpp
size_t strlen(const char* str) {
  const char* char_ptr;
  for (char_ptr = str; char_ptr != NULL; ++char_ptr)
    if (*char_ptr == '\0') return char_ptr - str;
}
```
