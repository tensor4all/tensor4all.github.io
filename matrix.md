---
title: Matrix
---
# Join us on Matrix

Most tensor4all discussion happens in our **Matrix space**: about 50 members, from first time users to the core developers. It is where development is discussed in the open, new papers get shared and meetings are announced.

<div class="actions">
  <a class="btn primary" href="https://matrix.to/#/#tensor4all:matrix.org">Open the tensor4all space</a>
</div>

## Rooms

<ul class="rooms">
  <li><code>announcements</code><span>Meetings, releases and events. Low traffic; worth keeping notifications on.</span></li>
  <li><code>all</code><span>General discussion and questions about methods and software.</span></li>
  <li><code>tenferro</code><span>Development of tenferro-rs and the layers below it.</span></li>
  <li><code>papers</code><span>New papers on tensor networks, TCI and related methods.</span></li>
  <li><code>github-feed</code><span>A live feed of activity in the tensor4all repositories.</span></li>
</ul>

## Join in three steps

1. **Pick a client and create an account.** On the desktop, we recommend [Koushi](https://shinaoka.github.io/koushi-matrix/) (see below). [Element](https://element.io/download), the official client, runs in the browser and on mobile too. An account on matrix.org works, as does any other homeserver.
2. **Set up encryption recovery right away.** Matrix is end to end encrypted. Save your recovery key so you do not lose message history when you sign in on a new device. Our [setup checklist](https://shinaoka.github.io/blog/matrix-encryption-setup-checklist/) walks through it, and the [safety and recovery checklist](https://shinaoka.github.io/blog/matrix-encryption-safety-recovery-checklist/) covers what to do if something goes wrong.
3. **Join the space.** Open [#tensor4all:matrix.org](https://matrix.to/#/#tensor4all:matrix.org), join the space, then join the rooms you want.

## Why Matrix?

Matrix is an open, federated protocol: no single company owns the network, anyone can run a server, and history stays searchable instead of disappearing behind a paywall. The longer argument is in [Why Matrix](https://shinaoka.github.io/blog/why-matrix/).

## Koushi: a Matrix client from our community

Element is the reference Matrix client and runs everywhere. In daily use, though, we kept hitting the same rough edges: messages stuck at **"Unable to decrypt"**, encrypted history that is hard to search, and typing and searching in Japanese, Chinese and Korean. So H. Shinaoka started building **[Koushi](https://shinaoka.github.io/koushi-matrix/)**, a desktop client designed around how we actually talk.

- **Fewer decryption failures**: Koushi has you finish secure backup and save your recovery key at sign-in, before you start chatting. When a missing key arrives later, the affected messages appear on their own. If backup fails, a diagnostics view tells you whether the network or the server is at fault.
- **Multilingual by design**: IME-safe typing, CJK-aware search, and an interface in English, Japanese, Chinese and Korean.
- **Local full-text search**, including encrypted rooms.
- **All your accounts in one window**, each in its own tab with its own unread count.
- **Works with your AI assistant**: Claude Code, Codex and other MCP clients can search, summarize and reply through the running app, only for the accounts you allow.

Available for macOS, Windows and Linux. Questions and bug reports go to [#koushi-matrix:matrix.org](https://matrix.to/#/#koushi-matrix:matrix.org).

<div class="actions">
  <a class="btn" href="https://shinaoka.github.io/koushi-matrix/">Get Koushi</a>
</div>

Prefer email? Meeting announcements also go to the [mailing list](https://groups.google.com/g/tensor4all). See [Community](community.html) for everything else.
