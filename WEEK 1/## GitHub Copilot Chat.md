## GitHub Copilot Chat

- Extension Version: 0.27.3 (prod)
- VS Code: vscode/1.100.3
- OS: Windows

## Network

User Settings:
```json
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 20.207.73.85 (174 ms)
- DNS ipv6 Lookup: Error (14 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy URL: None (13 ms)
- Electron fetch (configured): HTTP 200 (238 ms)
- Node.js https: HTTP 200 (188 ms)
- Node.js fetch: HTTP 200 (170 ms)
- Helix fetch: HTTP 200 (313 ms)

Connecting to https://api.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.112.22 (15 ms)
- DNS ipv6 Lookup: Error (26 ms): getaddrinfo ENOTFOUND api.githubcopilot.com
- Proxy URL: None (16 ms)
- Electron fetch (configured): HTTP 200 (813 ms)
- Node.js https: HTTP 200 (892 ms)
- Node.js fetch: 