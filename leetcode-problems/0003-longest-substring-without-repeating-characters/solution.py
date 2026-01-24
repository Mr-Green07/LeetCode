class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last = {}          # char -> last index
        left = 0           # left end of window
        best = 0

        for right, ch in enumerate(s):
            if ch in last and last[ch] >= left:
                left = last[ch] + 1      # move left past previous same char
            last[ch] = right
            best = max(best, right - left + 1)

        return best

