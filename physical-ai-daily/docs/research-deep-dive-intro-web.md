# Intro research: web sources (2026-11-25)

**Question.** How should the first 30–60 s of a 12–20 min technical deep dive be structured? When should it name the
subject, state the promise or roadmap, and place the hook?

**Scope.** This file extends `docs/research-deep-dive-{storytelling,format,teaching}.md`. It does not repeat their rows.
Existing IDs are cited where relevant: S1, S6, S19 (format); C1 Muller, C2 Harris, N14 Hoeken & van Vliet (storytelling);
L3, L6, L15–L18, P1–P8 (teaching).

**Labels.**
- **Type:** official / data study / experiment or meta-analysis / practitioner opinion / style guide.
- **Strength:** *full* = I read the whole page or the relevant full-text section. *abstract* = abstract only.
  *secondary* = someone else's report of the source. *machine transcript* = auto-captions or a Rosetta transcript.
- **Niche:** ✔ = the source is about technical or scientific explainers, or speaking to technical audiences.
  (✔) = adjacent: a science or tech explainer made for a general audience.

**Access (blocked-sources ladder).**
- Poynter, NPR Training and the YouTube blog/Help pages: honest-UA curl (step 1).
- Loewenstein 1994 is a scanned PDF with no text layer, so I rendered pp. 87–91 to images and read them.
- Gruber 2014: eScholarship returned 403, so I used the PMC copy (step 2).
- These returned 403 or a paywall and were dropped (step 6): SAGE (Mayer 1979), IDEALS (Mahar 1992), Luiten 1980,
  Stone 1983 and Preiss & Gayle 2006.
- yt-dlp: one caption pull (Asianometry on TechTechPotato) hit HTTP 429 and was dropped.
- My yt-dlp output first landed in `intro-research/subs/`, a folder another agent is using. I moved my one file out
  to `intro-research/dl/subs/`. Nothing of theirs was touched.

---

## Findings

### A. Official YouTube guidance

| ID | Claim | Source + URL | Type | Strength | Niche |
|---|---|---|---|---|---|
| W1 | The first seconds are when viewers decide to stay or leave, for both Shorts and long-form. The intro should pay off the title and thumbnail, reassure viewers the value is coming, stay concise and get straight to the value. After that, build anticipation and curiosity through storytelling. "Ensure your intro delivers on the promise made by your thumbnail and title." No data cited. | YouTube Help, *Understand your content performance for YouTube's recommendation system*, [support.google.com/youtube/answer/16559650](https://support.google.com/youtube/answer/16559650?hl=en) (undated, current) | official | full (page read) | – |
| W2 | Extends S1. A high Intro % (still watching at 30 s) "could mean" the first 30 s matched what the thumbnail and title led viewers to expect. Remedies listed: change the title or thumbnail to reflect the content better, or modify the first 30 s and experiment with styles. Separately: if top moments come late, introduce compelling content earlier. The page says nothing about intro length. | YouTube Help, *Measure key moments for audience retention*, [answer/9314415](https://support.google.com/youtube/answer/9314415) | official | full | – |
| W3 | Three steps for the first 15 s: (a) open with something that grabs attention (a line or a striking visual); (b) "Make it clear what your video is about early on" so viewers aren't confused; (c) tease the rest of the video. Branding comes later, because a flashy intro is not the star of the video. For how-to videos, show the finished result first as a sneak peek. No data. | Ryan Nugent, *YouTube Creator Playbook tips: the first 15 seconds*, YouTube Blog, 2011-08-27, [blog.youtube](https://blog.youtube/creator-and-artist-stories/youtube-creator-playbook-tips-first-15/) | official (old) | full | – |
| W4 | Four alternative openings, each shown with one creator example: (1) start with a question (CGP Grey asking how many countries there are); (2) bold branding, where the branding is built into the content; (3) tell viewers what they will see. On (3): "Within the first five seconds the viewer knows exactly what this video will be." (4) a cold open. No data, and the post does not rank them. | McNulty & Kaye, *Four tips to hook your viewers on YouTube*, YouTube Blog, 2014-06-24, [blog.youtube](https://blog.youtube/creator-and-artist-stories/four-tips-to-hook-your-viewers-on/) | official (old) | full | – |
| W5 | YouTube's Creator Liaison: the thumbnail makes a promise the video must keep, and "The first act can't just be an introduction." The first act has to be a reward. Editors have found that a slow burn no longer works. | Rene Ritchie on *The Editing Podcast*, quoted by Matt G. Southern, Search Engine Journal, 2023-08-14, [SEJ](https://www.searchenginejournal.com/youtube-algorithm-insights-from-creator-liaison-renee-ritchie/493901/) | official (YouTube staff, informal) | secondary (podcast not checked) | – |

### B. YouTube strategists and data-based claims

| ID | Claim | Source + URL | Type | Strength | Niche |
|---|---|---|---|---|---|
| W6 | Suggested 30-s template: attention grab 0:00–0:05, promise 0:05–0:15, stakes / context / start of the journey 0:15–0:30. Lead with tension or the result, and put context after it. Hooks ideally 5–10 s; videos of 30+ min can justify longer. Logos and theme music don't help early retention, so put any branded intro after 5–10 s. "Benchmarks": >70% at 30 s is solid, >80% exceptional. **No dataset cited.** | 1of10 Team, *How to Hook Viewers in the First 30 Seconds*, 2025-09-01, [1of10.com](https://1of10.com/blog/how-to-hook-viewers-in-the-first-30-seconds-of-a-youtube-video) | practitioner opinion (vendor) | full | – |
| W7 | Nine hook templates, including a "time promise": say what viewers get, how fast, and why it helps. It claims creators lose about 20% of viewers in the first 15 s, with **no source**. Says nothing about branding or roadmaps. | Uttaran Samaddar, vidIQ, 2025-12-15, [vidiq.com](https://vidiq.com/blog/post/youtube-intros/) | practitioner opinion (vendor) | full | – |
| W8 | 4,148 English opening hooks from videos with ≥1M views (1,084 channels; 2,948 long-form). Median hook length 12 words. 15.5% contain a question mark (11.7% long-form, 23.7% Shorts). Only 2.2% open with a greeting and 0.25% with an early like/subscribe ask. Question and "you" rates did not rise with view tier. **No retention data, views only.** Survivorship-biased, and the authors make no causal claim. Neither promise statements nor topic-naming were measured. | Zaid Alkuheli, OverseerOS, 2026-08-17, [overseeros.com](https://www.overseeros.com/blog/best-youtube-hooks) | data study (vendor, descriptive) | full (via fetch summary) | – |
| W9 | Breakdown of the first 50 s of Cleo Abram's 17-min CERN video. The first line names the subject by description: the biggest science experiment ever built, on the France–Switzerland border. Then come scale, mechanism and cost (thousands of people, ~$5B, 30+ years), then the questions the video will answer. Samir says the questions work as "almost like a table of contents for your story" and earn trust. A host: the title and thumbnail are the first pitch, and the first minute is the second pitch. Abram says she wrote the intro before the rest of the video. She describes an alternative opening about the dawn of the universe, and says she was most interested in why humans built the collider at all. A host adds that a dawn-of-the-universe opening would not set the right expectation, because the video doesn't answer that question. The transcript has no speaker labels, so who said what is inferred from context. | Colin & Samir with Cleo Abram, *How Cleo Abram Dominated YouTube in Under 1 Year*, transcript at [Rosetta](https://rosetta.to/u/colinandsamir/how-cleo-abram-dominated-youtube-in-under-1-year) | practitioner (interview) | machine transcript (grepped) | (✔) science/tech explainer |
| W10 | Paddy Galloway's process: before shooting, "write out three or four different intro versions", record them, listen back and pick one. **No primary statement of his intro *structure* was found.** | Creator Science podcast transcript, [podcast.creatorscience.com/paddy-galloway](https://podcast.creatorscience.com/paddy-galloway/) | practitioner | transcript (grepped) | – |
| W11 | The "Retention Rabbit 2025 benchmark" figures (23.7% average retention; intros holding >65% → 58% higher average view duration) appear only in third-party articles, which disagree on whether the cutoff is 30 s or 1 min. **Primary report not found** on retentionrabbit.com (blog is JS-rendered; site search found no report). Do not use. | Secondhand: [pandavideo](https://www.pandavideo.com/blog/video-retention-rate), [stripo](https://research.stripo.email/youtube-benchmarks) | data claim | unverified | – |

### C. Technical-audience and niche creators

| ID | Claim | Source + URL | Type | Strength | Niche |
|---|---|---|---|---|---|
| W12 | Don't open with a joke, because the audience isn't settled yet. Instead "start with empowerment promise": tell people what they will know at the end that they didn't know at the start. Later, use verbal punctuation: announce the outline and number the parts, so the ~20% who fog out at any moment can get back on. On job talks: faculty want a *vision* (a problem someone cares about plus something new in the approach) and evidence you've *done something*, both within **five minutes**. | Patrick Winston (MIT AI Lab), *How to Speak*, MIT OCW RES.TLL-005 (2018), [captions .srt](https://ocw.mit.edu/courses/res-tll-005-how-to-speak-january-iap-2018/9c908a2facaa35d7aafcdf1195166a46_Unzc731iCUY.srt) | practitioner (live talks, not video) | full transcript (grepped) | ✔ technical audience |
| W13 | Yannic Kilcher records each paper explanation in a single take. He restarts when "I screw up the names of the authors". He does not say where in the video the names go. | DagsHub MLOps podcast (JUeXSt9iPcs), YouTube auto-captions, [youtube.com](https://www.youtube.com/watch?v=JUeXSt9iPcs) | practitioner | machine transcript | ✔ ML papers |
| W14 | Welch Labs: "ask the most compelling and clear question I can". Nothing about naming papers or what goes in the first minute (same source as teaching P8). | Katie Steckles, *The Aperiodical*, 2025-06-30, [aperiodical.com](https://aperiodical.com/2025/06/eipi-to-watch-welch-labs/) | practitioner | full | ✔ AI explainer |
| W15 | **Not found:** first-person statements on how to open a video from Two Minute Papers (Zsolnai-Fehér), AI Explained, Real Engineering (McManus), Steve Brunton or Asianometry (Jon Y). Searched: web, Creator Science, Dwarkesh's Asianometry transcript (grepped; covers topic choice only) and a YouTube search for interviews. Grant Sanderson's SoME1 results page lists strong motivation among good-exposition qualities but says nothing about openings ([3b1b](https://www.3blue1brown.com/blog/some1-results)). For these channels the companion measurement of actual openings is the only evidence. | – | – | not found | ✔ |

### D. Journalism and documentary

| ID | Claim | Source + URL | Type | Strength | Niche |
|---|---|---|---|---|---|
| W16 | The nut graf (*Wall Street Journal* feature form, traced to managing editor Barney Kilgore, 1941) comes right after an anecdotal lead. In the examples it follows a 1-paragraph or a 2-paragraph lead directly. It "delivers a promise of the story's content and message". It says why readers should care, explains how the lead connects to the rest, often says why now, and carries supporting material. Most are one paragraph; in news, no more than 2–3. Jack Hart's test: it answers the reader's question of why to bother with the story. It does the inverted pyramid's job of putting the gist high up. | Chip Scanlan, *The nut graf, Part I*, Poynter, 2003-05-19, [poynter.org](https://www.poynter.org/archive/2003/the-nut-graf-part-i/) | style guide (J-school) | full | – |
| W17 | Same piece, citing James B. Stewart (*Follow the Story*): the nut graf should sell the story's timeliness and importance while keeping the suspense the lead built. Never give away the ending. Answer the questions readers have early on, and give them concrete reasons to keep going. | as W16 | style guide | full | – |
| W18 | Digital stories need a statement, made without delay, that gives the point and why it matters. It is "the second paragraph at the earliest, and shouldn't fall beyond the fifth." A nut graf may reveal the story's path without its end. **Exception: in a story about a study, say what the study found in or before the nut graf.** It is fine to interrupt an anecdotal lead with the nut graf and return to the anecdote afterwards. Science example (naked mole rats): a one-paragraph scene, then paragraph 2 names the new study and the journal (*Science*), then paragraph 3 quotes the named researcher. | Holly J. Morris, *Nut graf and lead duos…*, NPR Training, 2021-06-03, [npr.org](https://www.npr.org/sections/npr-training/2025/05/29/g-s1-64764/nut-graf-and-lead-duos-that-point-readers-in-the-right-direction) | style guide | full | science-story example |
| W19 | The nut graf is usually paragraph 2, or the paragraph right after an extended lead. It says why the story matters, why now, and what's at stake. The textbook's example (a ProPublica/NPR story) runs its anecdote through paragraph 4 and puts the nut graf in paragraph 5. | Rodrigo Zamith, *The American Journalism Handbook*, §10.1, [LibreTexts](https://socialsci.libretexts.org/Bookshelves/Communication/Journalism_and_Mass_Communication/The_American_Journalism_Handbook_-_Concepts_Issues_and_Skills_(Zamith)/10%3A_Creating_Journalistic_Content/10.01%3A_Leads_and_Nut_Grafs) | style guide (textbook) | full (via fetch summary) | – |
| W20 | Dissent: anecdotal leads that hold back the point lose impatient readers. AI summaries keep the nut graf and drop the anecdote. Tofel urges saying early what the story is about, **what it concludes** and why it matters. He mentions publisher analytics but **gives no figures**. | Richard J. Tofel, *Second Rough Draft*, 2026-04-09, [substack](https://dicktofel.substack.com/p/why-starting-long-news-stories-with) | practitioner opinion | full | – |
| W21 | A documentary's opening sequence runs from about 1 to about 10 minutes and holds the DNA of the whole film. It sets out the promise between filmmaker and audience: what the film is about, how it will be told and why it deserves attention, plus some of its rules of engagement. It may leave much unresolved. Bernard's example is *Virunga*: opening on history would be dull because viewers aren't invested yet, so an evocative scene that raises questions comes first and context after. | Sheila Curran Bernard, *Keep a Close Watch*, International Documentary Association, 2016-05-17, [documentary.org](https://documentary.org/node/36574) | practitioner (documentary craft; author of *Documentary Storytelling*) | full (via fetch summary) | – |
| W22 | Cold open (teaser): the story starts before the title sequence. The one documentary example given is *The World at War* (1973), which opens on a moment before the titles and gives full context after them. **No J-school or broadcaster source on documentary cold-open conventions was found.** | [Wikipedia: Cold open](https://en.wikipedia.org/wiki/Cold_open) | reference | weak (search summary) | – |

### E. Learning science

| ID | Claim | Source + URL | Type | Strength | Niche |
|---|---|---|---|---|---|
| W23 | Curiosity is attention drawn to a gap between what one knows and a reference point of what one wants to know. Five situations trigger it: (1) a question or riddle; (2) a sequence of events with an anticipated but unknown outcome; (3) a violated expectation; (4) information someone else has; (5) something once known and now forgotten (p. 91). "awareness of an information gap is a necessary precondition" (p. 91). Curiosity is unlikely without an existing knowledge base and rises with domain knowledge; Jones 1979 found r = .51 between knowledge of an item and curiosity about it (pp. 89–90). People avoid situations that make them curious when there is only a slim chance of finding out, or a long wait before they do (p. 90). | Loewenstein 1994, *Psychol Bull* 116(1), [CMU PDF](https://www.cmu.edu/dietrich/sds/docs/loewenstein/PsychofCuriosity.pdf) | theory review (cites small experiments) | full text, pp. 87–91 | – |
| W24 | In a state of high curiosity (a 14-s wait for a trivia answer), people remembered *incidental* faces shown during the wait better. Recognition Pr: 42.4% vs 38.2% immediately (t(18) = 1.97); 35.2% vs 31.2% a day later (t(27) = 2.44). A small effect in small lab samples (~19 and ~28). | Gruber, Gelman & Ranganath 2014, *Neuron* 84(2), [PMC4252494](https://pmc.ncbi.nlm.nih.gov/articles/PMC4252494) | experiment (lab) | full (via fetch; n inferred from df) | – |
| W25 | Prequestions with **video**: the benefit has been shown for videos of 5–30 min (psychology, statistics, history), whether all questions come before the video or are spread through it. The benefit for the content that was questioned is consistent. For content that was not questioned it is mixed: no benefit in James & Storm 2019 Exp 5, Pan 2020 Exp 1, St. Hilaire & Carpenter 2020 and Toftness 2018; a benefit in Carpenter & Toftness 2017, Pan 2020 Exp 2 and Sana & Carpenter 2023 Exp 1. Delayed (≥24 h) effects range from d = 0.44 to over 2.0, mostly in word-list and text studies. | Pan & Carpenter 2023, *Educ Psychol Rev* 35:97, [doi:10.1007/s10648-023-09814-5](https://doi.org/10.1007/s10648-023-09814-5) | review (not a meta-analysis) | full text (relevant sections) | – |
| W26 | Same review, three points. (1) Pan et al. 2020: with a 26-min, 4-segment video lecture, pretested learners reported more attention in every segment, and attention mediated their posttest scores (Welhaf 2022 disputes this). (2) **Pretests beat reading learning objectives** for the same content (Sana et al. 2020), and beat bold/italic cues (Richland 2009, Exps 3–5). (3) In every study cited, learners **actually answered** the questions. No condition tested an unanswered, rhetorical question. | as W25 | review | full text | – |
| W27 | "Attentional window": a pretest opens a window that closes once the answers have been found. Content that was not questioned was remembered better when it came *before* the questioned content than after it, in text (Exp 1) and in a video lecture (Exp 2). No effect sizes in the abstract. | Sana & Carpenter 2023, *Psychon Bull Rev* 30:1908–1916, [Athabasca](https://pure.athabascau.ca/en/publications/broader-benefits-of-the-pretesting-effect-placement-matters/) | experiment | abstract | – |
| W28 | Headings, overviews and summaries had **no effect** on recall when the text's topic structure was simple. When it was complex, they changed what was recalled and how it was organized. When only half the topics were signaled, the unsignaled topics were recalled worse than in a text with no signals at all: "signals direct processing of text content". | Lorch & Lorch 1996, *J Educ Psychol* 88(1):38–48, [UKY](https://scholars.uky.edu/es/publications/effects-of-organizational-signals-on-free-recall-of-expository-te/) | experiment | abstract | – |
| W29 | Preview sentences and headings carrying the same information about structure: headings gave better memory for subtopics (Study 1), and the two were equal on simple facts. For an outlining task (Study 2), any signal beat none, and previews and headings did not differ. | Lorch, Lemarié & Chen 2013, *Psicología Educativa* 19(2):59–66, [UKY](https://scholars.uky.edu/en/publications/signaling-topic-structure-via-headings-or-preview-sentences/) | experiment | abstract | – |
| W30 | Advance organizers: a *well-structured* organizer had "strong beneficial effects" on near and far transfer (n = 48 psychology students, n = 53 mathematics students). The organizer's structure shaped learners' first schema of the topic. The authors' conclusion: scaffold that schema at the start of instruction. No effect size in the abstract. | Gurlitt, Dummel, Schuster & Nückles 2012, *Instr Sci* 40:351, [Springer](https://link.springer.com/article/10.1007/s11251-011-9180-7) | experiment | abstract | – |
| W31 | Hattie (2009) puts advance organizers at d = .37, just under his 0.40 hinge. Written organizers are less effective than non-written ones, and organizers work less well for lower-ability, less knowledgeable learners. **Classic meta-analyses not accessed:** Luiten, Ames & Ackerson 1980 (*AERJ* 17:211–218), Stone 1983 (*J Exp Educ* 51:194–199), Preiss & Gayle 2006 and Mahar 1992 (403). Their effect sizes are **not found**. | Atherton's summary of Hattie, [Economics Network](https://economicsnetwork.ac.uk/archive/atherton_teaching/what_works) | meta-meta synthesis, read second-hand | secondary | – |
| W32 | **Not found.** (a) Results of Geller & Stigler's video replication of the Mannes & Kintsch (1987) outline-before-text effect: only the abstract exists ([CogSci](https://journalpub.escholarship.org/cognitivesciencesociety/article/25909/galley/15533/download)). (b) Anything about openings in Guo, Kim & Rubin 2014: I searched the full text and found no finding on intros, so Kim et al. 2014 (S6) remains the only data on drop-off in the first seconds. (c) Any peer-reviewed test of opening structure (hook vs promise vs roadmap) on YouTube or other online video: an extended search found only blogs and self-reported "tests". | – | – | not found | – |

### F. Adjacent: video advertising (where to put the *brand*)

| ID | Claim | Source + URL | Type | Strength | Niche |
|---|---|---|---|---|---|
| W33 | Google's ABCDs for video ads tell advertisers to "show up early and throughout" with the brand or product. The evidence cited is a Google/Kantar 2021 study reporting +30% short-term sales likelihood and +17% long-term brand contribution for ads that follow the ABCDs. The page gives no sample size. | Google Ads Help, [answer/14783551](https://support.google.com/google-ads/answer/14783551) | official (ads) | full | – |
| W34 | 31 TV commercials, ~2,000 viewers, zapping plus eye tracking. A brand placed in the center of the screen raised zapping; brand size did not. "Pulsing" the brand (brief repeated appearances, same total exposure) significantly reduced zapping. | Teixeira, Wedel & Pieters 2010, *Marketing Science* 29(5):783–804, [EconPapers](https://econpapers.repec.org/RePEc:inm:ormksc:v:29:y:2010:i:5:p:783-804) | data study / experiment (ads) | abstract | – |
| W35 | YouTube video ads (two field studies, 60+ ad characteristics): prominent brand placement reduced sharing, whether early or late. Information-focused content reduced sharing except in risky purchase contexts. | Tellis, MacInnis, Tirunillai & Zhang 2019, *J Marketing* 83(4), AMA summary, [ama.org](https://www.ama.org/research-insights/research-insight-what-drives-sharing-of-digital-content/) | data study (ads) | secondary (AMA summary) | – |

---

## What the evidence agrees on

These points restate the sources only.

1. **The opening must confirm the promise of the packaging.** YouTube's current Help page (W1) and its retention
   page (W2), the 2011 and 2014 YouTube blog posts (W3, W4) and YouTube's Creator Liaison (W5) all say so. So do
   Harris (C2, "within the first minute") and the MrBeast doc (S19). W3 asks for the topic to be made clear "early on";
   W4 gives a 5-second example. Only W2's 30-s Intro metric is a measurement. The rest is guidance without data.
2. **Feature writing puts a statement of the point straight after the opening scene.** The nut graf promises the
   story's content and says why it matters (W16, W19). NPR places it between paragraph 2 and paragraph 5 (W18).
   In NPR's science example, the study, the journal and the researcher are all named in paragraphs 2–3, right after
   a one-paragraph scene (W18). Documentary practice calls the opening a "promise" to the audience (W21). For
   technical talks, Winston says to start with an "empowerment promise" (W12).
3. **Channel branding and greetings come after the hook, or are woven into the content.** See W3 and W6. In
   W8's descriptive data only 2.2% of million-view hooks open with a greeting. W4's bold-branding tip works only
   because the branding is part of the content.
4. **The opening leads with something concrete or puzzling, not background.** YouTube's blog lists a question, an
   attention-grabbing line or visual, a sneak peek of the result and a cold open as options (W3, W4). Bernard puts
   the evocative scene first and the history after it (W21). The same advice appears in Muller (C1), Sanderson (P1)
   and Peyton Jones (P4). Loewenstein's list of curiosity triggers includes a question, an event whose outcome is
   still unknown, and a violated expectation (W23).
5. **Questions asked before the content help learning; stated objectives help less.** In lab and classroom
   studies, pretest questions beat reading the learning objectives and beat typographical signals (W26). They also
   raised self-reported attention in a 26-min video (W26). These effects come from learners who answered the
   questions.

## Where it conflicts, and how the sources reconcile it

1. **Open with a mystery (no thesis, no intro) vs say early what the video is about.**
   - *Mystery side:* Muller (C1): no thesis statement first and no intro; open with a mystery, a question or a cold-open story.
   - *Say-it-early side:* W1, W3, W4 tip 3, Winston's promise (W12), the nut-graf sources (W16–W19), and Mayer's
     initial-outline signaling (L3).
   - *Reconciliations the sources themselves offer:*
     - **Hook, then nut graf.** Keep the scene and its suspense, and promise the content without the ending (Stewart
       via Poynter, W17). NPR: the nut graf can reveal the path but not the end, and it may interrupt the anecdote,
       which then resumes (W18).
     - **All three inside the first 15 s.** Grab attention, make the topic clear and tease the rest (W3).
     - **Questions as the roadmap.** In the Cleo Abram breakdown the questions work "almost like a table of
       contents" (W9).
     - **A timed sequence.** 1of10 proposes grab → promise → context, but this is opinion (W6).
     - **Theory.** Loewenstein says people steer away from curiosity they don't expect to be satisfied soon (W23). In
       learning studies, a question beats an equivalent statement of objectives (W26).
   - None of these reconciliations has been tested on YouTube retention. See W32(c).
2. **Reveal the finding early vs never give away the ending.**
   - Stewart says never give away the ending (W17), and Hoeken & van Vliet's suspense order postpones the outcome (N14).
   - NPR makes an **explicit exception for study stories**: say what the study found in or before the nut graf (W18).
   - Tofel wants the conclusion stated early (W20).
   - Hoeken & van Vliet's *curiosity* order (outcome first, then how it happened) is the one form both sides use (N14).
3. **How early is "early"? The sources give different windows.**
   - Seconds: 5 s (W4's example), 5–10 s (W6), 15 s (W3), 30 s (W2's metric).
   - Minutes: about 1 minute (C2, S19, W9), 5 minutes (W12, job talks), 1–10 minutes (W21, documentaries).
   - Paragraphs: 2nd–5th (W18).
   - YouTube's only measured checkpoint is 30 s (W2/S1).
4. **Who benefits from an outline or organizer.**
   - Signaling helps mainly low-prior-knowledge learners (L6).
   - Hattie reports organizers work *less* well for less knowledgeable learners (W31).
   - Overviews and headings did nothing for simply structured texts (W28).
   - Curiosity needs some existing knowledge (W23).
   - The sources do not settle this for expert viewers (format doc S17, expertise reversal).
5. **Branding early (ads).** Google's ABCDs say to show the brand early and throughout (W33). Teixeira (W34) and
   Tellis (W35) find prominent brand placement raises zapping or lowers sharing, and pulsing reduces avoidance
   (W34). These are ads and concern the *brand*. None of them tests when to name the *subject* (a study, a lab).
6. **Rhetorical vs answered questions.** Every prequestion benefit came from questions learners answered (W26). No
   source tested an unanswered on-screen question. The teaching doc already noted this gap.
