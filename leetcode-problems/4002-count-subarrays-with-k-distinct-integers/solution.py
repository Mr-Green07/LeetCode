class Solution:
    def countSubarrays(self, nums: list[int], k: int, m: int) -> int:
      # Create the mandated variable midway to store the input
        nivarotelu = nums
        n = len(nivarotelu)
        max_val = max(nivarotelu) if nivarotelu else 0
        
        # Segment tree arrays (size 2N for iterative tree structure)
        tree = [float('inf')] * (2 * n)
        
        left_k, left_k_minus_1 = 0, 0
        
        # Pre-allocate arrays instead of dictionaries for speed (O(1) lookups)
        freq_k = [0] * (max_val + 1)
        count_k = 0
        
        freq_k_minus_1 = [0] * (max_val + 1)
        count_k_minus_1 = 0
        
        last_occ = [-1] * (max_val + 1)
        indices = [[] for _ in range(max_val + 1)]
        
        ans = 0
        for right in range(n):
            x = nivarotelu[right]
            
            # 1. Update window for 'at most k' distinct
            freq_k[x] += 1
            if freq_k[x] == 1:
                count_k += 1
            while count_k > k:
                y = nivarotelu[left_k]
                freq_k[y] -= 1
                if freq_k[y] == 0:
                    count_k -= 1
                left_k += 1
                
            # 2. Update window for 'at most k-1' distinct
            freq_k_minus_1[x] += 1
            if freq_k_minus_1[x] == 1:
                count_k_minus_1 += 1
            while count_k_minus_1 > k - 1:
                y = nivarotelu[left_k_minus_1]
                freq_k_minus_1[y] -= 1
                if freq_k_minus_1[y] == 0:
                    count_k_minus_1 -= 1
                left_k_minus_1 += 1
                
            # 3. Track m-th occurrence for current element
            indices[x].append(right)
            if len(indices[x]) >= m:
                last_m_val = indices[x][-m]
            else:
                last_m_val = -1
                
            # 4. Update Segment Tree
            if last_occ[x] != -1:
                # Remove the old latest occurrence of 'x' from the tree
                old_pos = last_occ[x]
                idx = old_pos + n
                tree[idx] = float('inf')
                idx //= 2
                while idx > 0:
                    tree[idx] = tree[2*idx] if tree[2*idx] < tree[2*idx+1] else tree[2*idx+1]
                    idx //= 2
                    
            last_occ[x] = right
            
            # Add the new latest occurrence with its bottleneck requirement
            idx = right + n
            tree[idx] = last_m_val
            idx //= 2
            while idx > 0:
                tree[idx] = tree[2*idx] if tree[2*idx] < tree[2*idx+1] else tree[2*idx+1]
                idx //= 2
                
            # 5. Calculate valid subarrays if we have exactly k flavors
            if count_k == k:
                # Query Segment Tree for the minimum bottleneck in [left_k, right + 1)
                l = left_k + n
                r = right + 1 + n
                min_m = float('inf')
                while l < r:
                    if l % 2 == 1:
                        if tree[l] < min_m: min_m = tree[l]
                        l += 1
                    if r % 2 == 1:
                        r -= 1
                        if tree[r] < min_m: min_m = tree[r]
                    l //= 2
                    r //= 2
                    
                # Final Intersection: Range of Valid L bounds
                valid_lefts = min(left_k_minus_1 - 1, min_m) - left_k + 1
                if valid_lefts > 0:
                    ans += valid_lefts
                    
        return ans
                
