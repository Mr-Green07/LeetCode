class Solution:
    def change(self, amount: int, coins: list[int]) -> int:
        # Initialize dp array with 0s, size amount + 1
        dp = [0] * (amount + 1)
        
        # Base case: 1 way to make change for 0 amount (using no coins)
        dp[0] = 1
        
        # Process each coin one by one to ensure combinations, not permutations
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
                
        return dp[amount]
