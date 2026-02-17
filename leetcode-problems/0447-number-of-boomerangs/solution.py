class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        res = 0
        
        for p in points:
            dist_map = {}
            
            for q in points:
                # Calculate squared Euclidean distance
                # (x1-x2)^2 + (y1-y2)^2
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                dist = dx * dx + dy * dy
                
                # Update frequency map
                dist_map[dist] = dist_map.get(dist, 0) + 1
            
            # Calculate permutations for each distance group
            for count in dist_map.values():
                res += count * (count - 1)
                
        return res
