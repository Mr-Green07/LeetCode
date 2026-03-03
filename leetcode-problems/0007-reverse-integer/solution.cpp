#include <climits> // For INT_MAX and INT_MIN

class Solution {
public:
    int reverse(int x) {
        int rev = 0;
        
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            
            // Check for positive overflow before multiplying by 10
            // INT_MAX is 2147483647. So INT_MAX/10 is 214748364.
            if (rev > INT_MAX / 10 || (rev == INT_MAX / 10 && pop > 7)) {
                return 0; 
            }
            
            // Check for negative underflow before multiplying by 10
            // INT_MIN is -2147483648. So INT_MIN/10 is -214748364.
            if (rev < INT_MIN / 10 || (rev == INT_MIN / 10 && pop < -8)) {
                return 0;
            }
            
            rev = rev * 10 + pop;
        }
        
        return rev;
    }
};

