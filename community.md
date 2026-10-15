---
title: Community
---
# Community

tensor4all is an open community. Anyone working on tensor networks, tensor cross interpolation or the Rust stack is welcome, from first questions to pull requests.

## Where we talk

<div class="cards">
  <div class="card feature">
    <span class="tag user">chat</span>
    <h3>Matrix space</h3>
    <p>Our main place for discussion, with separate rooms for announcements, papers, tenferro development and a live GitHub feed.</p>
    <a class="go" href="matrix.html">How to join &rarr;</a>
  </div>
  <div class="card">
    <span class="tag user">announcements</span>
    <h3>Mailing list</h3>
    <p>Meeting announcements and Zoom links, 120+ subscribers. Without a Google account, write to <a href="mailto:tensor4all-admin@googlegroups.com">tensor4all-admin@googlegroups.com</a>.</p>
    <a class="go" href="https://groups.google.com/g/tensor4all">Subscribe &rarr;</a>
  </div>
  <div class="card">
    <span class="tag user">code</span>
    <h3>GitHub</h3>
    <p>Issues and pull requests on any repository are welcome. Each repository's README and AGENTS.md describe how to contribute.</p>
    <a class="go" href="https://github.com/tensor4all">github.com/tensor4all &rarr;</a>
  </div>
</div>

The [Discourse forum](https://tensor4all.discourse.group) is kept in read-only mode as an archive.

## Monthly online meetings {#meetings}

We meet online about once a month. Developer meetings cover progress on the software and open design questions; talks are given by invited speakers. The Zoom link is sent to the [mailing list](https://groups.google.com/g/tensor4all) before each meeting.

### Upcoming

{% assign today = site.time | date: '%Y-%m-%d' %}
<ul class="meetings" data-upcoming-list="comm-up">
{% assign upcoming = site.data.meetings | reverse %}{% for m in upcoming %}{% assign d = m.date | date: '%Y-%m-%d' %}{% if d >= today %}{% include meeting.html m=m upcoming=true past_target="past-list" %}{% endif %}{% endfor %}
</ul>
<p id="comm-up-none" hidden>The next date will be announced on the mailing list and in the Matrix announcements room.</p>

### Past meetings

<ul class="meetings" id="past-list">
{% for m in site.data.meetings %}{% assign d = m.date | date: '%Y-%m-%d' %}{% if d < today %}{% include meeting.html m=m %}{% endif %}{% endfor %}
</ul>

## How we build

Much of the Rust stack is developed with AI coding agents under human review. Two things we share openly:

- **Workflow.** [Agentic AI coding in Rust](https://shinaoka.github.io/agentic-ai-coding-rust/slides_en.pdf) (slides) describes the rules, review and testing behind tenferro-rs and tensor4all-rs.
- **Provenance.** After a developer meeting on copyright and scientific credit in July 2026, we adopted [provenance rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/provenance.md) for agents and per repository citation policies: [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs/blob/main/docs/PROVENANCE_AND_CITATION_POLICY.md), [tenferro-rs](https://github.com/tensor4all/tenferro-rs/blob/main/docs/PROVENANCE_AND_CITATION_POLICY.md), [strided-rs](https://github.com/tensor4all/strided-rs/blob/main/docs/PROVENANCE_AND_CITATION_POLICY.md).

## Workshops and people

We also organize in-person [workshops](workshop/index.html). See [About](about.html) for the people involved.
