class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        total_count = 0
        # List of dictionaries. dp[i][diff] stores count of subseqs ending at i with diff
        dp = [defaultdict(int) for _ in range(n)]
        
        for i in range(n):
            for j in range(i):
                diff = nums[i] - nums[j]
                
                # Get the number of sequences ending at j with this diff
                # These are sequences of length >= 2
                count_at_j = dp[j][diff]
                
                # If we extend a sequence ending at j, it becomes length >= 3
                # So we add those to our total result
                total_count += count_at_j
                
                # Update dp[i][diff]:
                # We add 'count_at_j' (extended sequences)
                # plus 1 (the new sequence [nums[j], nums[i]] of length 2)
                dp[i][diff] += count_at_j + 1
                
        return total_count
