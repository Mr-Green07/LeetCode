#include <string>
#include <climits>

using namespace std;

class Solution {
public:
    int myAtoi(string s) {
        int i = 0, n = s.length();
        int sign = 1;
        int result = 0;
        
        // 1. Ignore leading whitespace
        while (i < n && s[i] == ' ') {
            i++;
        }
        
        // 2. Check for sign
        if (i < n && (s[i] == '+' || s[i] == '-')) {
            sign = (s[i] == '-') ? -1 : 1;
            i++;
        }
        
        // 3. Process digits and build the number
        while (i < n && isdigit(s[i])) {
            int digit = s[i] - '0';
            
            // 4. Check for overflow BEFORE multiplying by 10
            // INT_MAX is 2147483647. INT_MAX / 10 is 214748364.
            if (result > INT_MAX / 10 || (result == INT_MAX / 10 && digit > 7)) {
                return (sign == 1) ? INT_MAX : INT_MIN;
            }
            
            result = result * 10 + digit;
            i++;
        }
        
        return result * sign;
    }
};
