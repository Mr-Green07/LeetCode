class Solution {
public:
    vector<int> remainingMethods(int n, int k, vector<vector<int>>& invocations) {
        vector<vector<int>> graph(n);
        for(const auto& it: invocations){
            graph[it[0]].push_back(it[1]);
        }

        vector<bool> suspicious(n, false);
        suspicious[k] = true;
        queue<int> q{{k}};
        while(!q.empty()){
            int size = q.size();
            while(size--){
                int a = q.front(); q.pop();
                for(int v : graph[a]){
                    if(!suspicious[v]){
                        suspicious[v] = true;
                        q.push(v);
                    }
                }
            }
        }
        for(int i = 0; i< n; ++i){
            if(suspicious[i]) continue;
            for(int v : graph[i]){
                if(suspicious[v]){
                    vector<int> all(n);
                    iota(all.begin(), all.end(), 0);
                    return all;
                }
            }
        }
        vector<int> ans;
        for(int i = 0; i< n; i++){
            
            if(!suspicious[i]){
                ans.push_back(i);
            }
        }
        return ans;

    }
};
