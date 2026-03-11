#include <string>
#include <algorithm>

using namespace std;

class Solution {
    int expandAroundCenter(const string& s, int left, int right) {
        while (left >= 0 && right < s.length() && s[left] == s[right]) {
            left--;
            right++;
        }
        // Return length of the palindrome found
        return right - left - 1;
    }

public:
    string longestPalindrome(string s) {
        if (s.empty() || s.length() < 2) {
            return s;
        }
        
        int start = 0;
        int max_len = 0;
        
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);     // Odd length
            int len2 = expandAroundCenter(s, i, i + 1); // Even length
            
            int current_max = max(len1, len2);
            
            if (current_max > max_len) {
                max_len = current_max;
                // Calculate the start index of the palindrome
                start = i - (current_max - 1) / 2;
            }
        }
        
        return s.substr(start, max_len);
    }
};
