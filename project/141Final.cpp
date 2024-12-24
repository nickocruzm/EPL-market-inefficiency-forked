#include <bits/stdc++.h>

using namespace std;

int main() {

    int numCity, numRoad;

    cin >> numCity >> numRoad;

    vector<int> sellPrices(numCity);
    for(auto& x : sellPrices) cin >> x;

    vector<vector<pair<int, int>>> edgeWeights(numCity);
    int i, j, w;
    for(int k = 0; k < numRoad; ++k) {
        cin >> i >> j >> w;
        edgeWeights[i].push_back({j, w});
        edgeWeights[j].push_back({i, w});
    }

    vector<pair<bool, int>> dijkstraHelper (numCity);
    for(int i = 0; i < numCity; ++i) {
        dijkstraHelper[i].first = false;
        dijkstraHelper[i].second = INT_MAX;
    }

    dijkstraHelper[0].second = 0;

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    pq.push({0, 0});


    while(!pq.empty()) {

        // O(1) to find top node
        auto [dist, current] = pq.top();
        pq.pop();

        // continue if already visited
        if (dijkstraHelper[current].first) continue; 
        dijkstraHelper[current].first = true;

        for (auto [neighbor, weight] : edgeWeights[current]) {
            if (dist + weight < dijkstraHelper[neighbor].second) {
                dijkstraHelper[neighbor].second = dist + weight;
                pq.push({dijkstraHelper[neighbor].second, neighbor});
            }
        }
    }

    int best = 0;

    for(int i = 0; i < numCity; ++i) {
        best = max(best, sellPrices[i] - 2*dijkstraHelper[i].second);
    }

    cout << best << endl;

    return 0;
}
