# Research 2026-11-15-weekly (window: 2026-11-08 → 2026-11-15)

Ranked by the judge panel: 6 passes (Opus, Sonnet and Fable, two card orders each), cut line settled pairwise in both orders. Cards: `judge/cards.md`. Everything below is quoted from stored sources; summaries and explanations are written at the scripting step.

## Shortlist

| # | Story | Kind | Panel | Sig | Pull | Evid | Show | Mom | Judges' lineups | Primary source (date) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | NVIDIA: GR00T 2 robot model | learning | 8.17 (spread 0.5) | 7.17 | 6 | 7.17 | 8 | 8 | 6/6 | https://research.nvidia.com/labs/gear/gr00t-2/ (2026-11-09) |
| 2 | Tesla: Cybercab offered for personal purchase; Robotaxi hours and San Antonio launch | news | 8.17 (spread 1.0) | 7.33 | 9.33 | 6 | 6.83 | 9.33 | 6/6 | https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/ (2026-11-13) |
| 3 | Google DeepMind: Gemini Robotics 2.5; Apptronik Apollo 2 pilot at Mercedes-Benz | learning | 8 (spread 0) | 7.17 | 6.83 | 6.5 | 8 | 8 | 6/6 | https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/ (2026-11-10) |
| 4 | UC Berkeley: SuperMimic humanoid control paper (CoRL 2026) | learning | 7.0 (spread 1.0) | 6.67 | 4.5 | 7 | 7.83 | 6 | 6/6 | https://arxiv.org/abs/2610.28480 (2026-11-12) |
| 5 | Anthropic: Claude Fable 5.2 robot-control results | learning | 6.5 (spread 1) | 6.67 | 6.67 | 5 | 6.5 | 7.33 | 4/6 | https://www.anthropic.com/claude-fable-and-mythos-5-2 (2026-11-09) |

Editorial rule (at least 2 learning stories): already met, nothing moved.

### 1. NVIDIA: GR00T 2 robot model

Status: **confirmed**. Primary: https://research.nvidia.com/labs/gear/gr00t-2/

**Source quotes:**
- "an open frontier world action model built on NVIDIA Cosmos" (https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/)
- "One diffusion transformer denoises future video and robot actions together." (https://research.nvidia.com/labs/gear/gr00t-2/)
- "GR00T 2 is pretrained on one mixture of three kinds of data: 402,730 hours of egocentric video of people at work," (https://research.nvidia.com/labs/gear/gr00t-2/)
- "Completed 40 of 50 evaluated tasks at least once without task-specific demonstrations." (https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/)
- "One demonstration per task lifts success to 71%, and five, under 3 minutes of robot data, lift it to 95%" (https://research.nvidia.com/labs/gear/gr00t-2/)
- "With the agent, GR00T 2 assembles the toy plane, the toy tractor, and the Hot Wheels car in 98, 97, and 97 of 100 trials." (https://research.nvidia.com/labs/gear/gr00t-2/)
- "a smaller joint video–action transformer (2B parameters)" (https://research.nvidia.com/labs/gear/gr00t-2/)
- "Trained on diverse robot and human data, including 100,000 hours of NVIDIA open source human egocentric data" (https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/)
- "Code Available in December" (https://research.nvidia.com/labs/gear/gr00t-2/)

**Not stated in the source:**
- Full GR00T 2 parameter count not stated (only GR00T 2 Turbo: 2B)
- Pretraining compute not stated
- Weights release date and license not stated; code labelled 'Code Available in December' with no year
- Research page shows 'November 2026' with no specific day
- Error bars / statistical detail only on the human-video scaling chart
- Sim-real correlation (Pearson 0.92) appears in a chart label only
- Blog gives no availability date for GR00T 2

**What people say** (from the card):
- X @nvidia (2,720,021 followers, 3,017 likes, 2026-11-13): "Meet GR00T 2, our most advanced, open System One model for robotics research. It learns skills from human demonstrations and predicts how its actions will change the physical world. Those skills can carry across different robot bodies, reducing the need to teach each robot every " https://x.com/nvidia/status/2121319866712076769
- X @WatcherGuru (4,923,742 followers, 2,879 likes, 2026-11-13): "JUST IN: Nvidia $NVDA unveils 'GR00T 2' AI model that allows robots to learn from humans and share skills with other robot bodies. https://t.co/oGpxZNR7sd" https://x.com/WatcherGuru/status/2121321121031377336
- X @AutismCapital (1,749,756 followers, 1,942 likes, 2026-11-13): "Jensen personally trained GR00T 2 https://t.co/MEGOZUBVTw" https://x.com/AutismCapital/status/2121378003418837284
- X @DrJimFan (605,188 followers, 1,911 likes, 2026-11-10): "Introducing GR00T 2: our most advanced, open System One model for robotics. It runs entirely on your robot’s onboard compute, fully untethered, and you own every weight of it. GR00T 2 features: 1. Long-horizon tasks: dexterous manipulation and whole-body control chained over 20+ " https://x.com/DrJimFan/status/2120186651741655428
- YouTube "Claude Fable 5.2 AI Robot Control, NVIDIA GR00T 2, Atlas Wheels, Figure AI" (AI News, 6,055 views) https://www.youtube.com/watch?v=zyM4QjcWp3c
- YouTube "A robot model that imagines the future: NVIDIA GR00T 2 explained" (paper2video, 691 views) https://www.youtube.com/watch?v=AftuizPLksQ
- YouTube "NVIDIA's GR00T 2: The Robot That Learns by Watching Humans | A ByteX Documentary" (ByteX, 54 views) https://www.youtube.com/watch?v=3PXaTnvF5dE

**What the judges wrote:**
- A1: "NVIDIA's GR00T 2 is an open world-action model with specific, concrete results: 71% success from one demo and 95% from five, and 97-98 of 100 trials on agentic assembly, trained on 402,730 hours of human video. The numbers come from NVIDIA alone, and the weights date and license are unclear. There are many demo videos and charts, and it got heavy engagement, including mainstream finance accounts."
- A2: "NVIDIA's research page gives specific numbers: 402,730 hours of human video, 95% success with five demonstrations, and 97-98 of 100 assembly trials. It also has many demo videos, and code is promised for December. It is an open world action model that learns from human video and moves across robot bodies. The field's main accounts covered it widely."
- B1: "Official NVIDIA research page with specific numbers (71% with one demo, 95% with five, 97-98 of 100 assembly trials) and many demo videos; results are self-reported and weights and licence dates are not stated (code due in December). A major open world-action model from the field's key supplier, with the strongest buzz of any model release. Easy to show."
- B2: "NVIDIA's research page and blog give specific numbers (402,730 hours of human video, 71% with one demo, 95% with five, 97-98 of 100 assembly trials) and rich demo videos. Open model for robot control is a real step, though weights and code arrive later and results are NVIDIA's own. Top engagement of the week."
- C1: "NVIDIA's research page gives concrete figures for an open world action model: 402,730 hours of human egocentric video, 40 of 50 tasks completed zero-shot, 95% with five demonstrations, and 97-98 of 100 agentic toy-assembly trials, with many demo videos and charts. Weights and license are not yet out ('Code Available in December') and the results are NVIDIA's own, so it is a strong announcement rather than a release in hand. Momentum is high (3,017 likes, 32 posts, 15 harvest items) and 'robots learn by watching people work' is an everyday analogy."
- C2: "NVIDIA's research page gives hard numbers for an open world-action model: pretrained on 402,730 hours of egocentric human video, 40 of 50 tasks completed zero-shot, 71% with one demonstration and 95% with five, and 97-98 of 100 agentic assemblies, backed by many demo videos and charts. Code is 'available in December' with no weights date or license, and the full parameter count and compute are unstated, so the release is a promise with a detailed preview. Momentum is high and reaches beyond robotics (3,017 likes from NVIDIA, 2,879 from WatcherGuru, 1,911 from Jim Fan, 6.9k YouTube views), and 'a robot that learns by watching people work' is an analogy our viewer gets."

**Visuals:** Research page: hero demo videos (toy truck assembly on Dexter); Architecture rollout video (head and two wrist cameras); Human-video scaling charts (log scale, 1 SD error bars); Videos: conveyor sorting, cable and charger insertion, toy plane, popcorn delivery on Unitree G1, hammer sim-to-real; Charts: zero-shot/few-shot averages, real-world RL, agentic assembly counts, latency, sim-real scatter plots; Blog: groot-2-shipping-label.mp4 video; CoRL promo image

**Links:**
- https://research.nvidia.com/labs/gear/gr00t-2/
- https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/
- https://x.com/DrJimFan/status/2120186651741655428
- https://x.com/NVIDIARobotics/status/2119794454190006428
- https://x.com/nvidia/status/2121319866712076769
- https://x.com/linkevin0/status/2120410553936224439
- https://x.com/DrJimFan/status/2120540201994846253
- https://x.com/HuggingPapers/status/2119703201473495086
- https://x.com/TheHumanoidHub/status/2119893860994384305
- https://x.com/clankrmedia/status/2119849193829199944
- https://x.com/WallStreetB0Ts/status/2121331329971405056
- https://x.com/DrJimFan/status/2119862216656838909
- https://x.com/XRoboHub/status/2120589025828618720
- https://x.com/rohanpaul_ai/status/2120281923725795357
- https://x.com/clankrmedia/status/2120524939493097538
- …and 2 more harvest items in <ep>/triage/verdicts_*.json

### 2. Tesla: Cybercab offered for personal purchase; Robotaxi hours and San Antonio launch

Status: **confirmed**. Primary: https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/

**Source quotes:**
- "Tesla will soon start selling its steering wheel-free Cybercab for personal use, the company announced on Friday." (https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/)
- "The Cybercab will cost $29,990, undercutting the Model 3 by $7,000 and making the golden two-seater Tesla's cheapest vehicle ever. According to a screenshot of the order page shared by Tesla, deliveries will begin as soon as next month." (https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/)
- "It will also require a $299-per-month autonomous driving subscription to operate, with the first month included for no charge." (https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/)
- ""Select customers are now being invited to purchase Cybercab for personal use," the company said in a tweet on its Robotaxi X account. Tesla said initial availability is limited." (https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/)
- "For example, there was no mention of whether customers would be able to add their cabs to the Robotaxi fleet and earn passive income, another longtime Musk promise." (https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/)
- "The federal government lets companies apply to roll out noncompliant vehicles, but those deployments are capped at 2,500 vehicles per year." (https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/)
- "The estimated purchase price is $29,990 before taxes, paired with a non-refundable $1,000 deposit and an estimated December 2026 delivery in Austin, Texas." (https://www.teslarati.com/tesla-cybercab-arrives-steep-ownership-fine-print/)
- "Late on November 13, Tesla announced that select customers were being invited to order a Cybercab for personal use." (https://www.teslarati.com/tesla-cybercab-arrives-steep-ownership-fine-print/)
- "Personal, non-commercial use only; ride-hailing, Robotaxi network participation, delivery, or rental requires separate written authorization or product enablement from Tesla." (https://www.teslarati.com/tesla-cybercab-arrives-steep-ownership-fine-print/)
- "That covers all eight of Tesla’s unsupervised Robotaxi markets: Austin, San Antonio, Houston, Fort Worth, Dallas, Miami, Orlando, and Tampa. It includes both Model Y robotaxis and Cybercabs, though Cybercab rides are still limited to Austin for now." (https://www.notateslaapp.com/news/4991/tesla-robotaxi-now-runs-247-in-every-city)

**Not stated in the source:**
- No source states a specific delivery day such as December 20; InsideEVs says deliveries begin 'as soon as next month' and Teslarati says 'an estimated December 2026 delivery in Austin, Texas'
- No number of invited customers or units available for personal purchase stated
- No date stated for when sales open to the general public
- InsideEVs states no mention of whether owners can add cars to the Robotaxi fleet
- No Tesla primary page (order page or Robotaxi X post) was fetched directly; Tesla's statements are as quoted by InsideEVs and Teslarati
- The Teslarati San Antonio URL (published 2026-11-09) covers the San Antonio launch; the Cybercab sale text appears in later Teslarati articles appended to that page

**What people say** (from the card):
- X @robotaxi (289,409 followers, 25,067 likes, 2026-11-13): "Select customers are now being invited to purchase Cybercab for personal use https://t.co/LDIkKuseAZ" https://x.com/robotaxi/status/2121384435241402737
- X @GregoryMcFadden (13,729 followers, 6,118 likes, 2026-11-14): "Marques already won the bet. He said, under $30,000 and the Cybercab is $29,990 with a required $300 a month subscription. The price is already over $30,000 with the mandatory subscription, and the long-term cost is much higher." https://x.com/GregoryMcFadden/status/2121607659501826427
- X @TheCaptainEli (30,068 followers, 2,697 likes, 2026-11-14): "I don't give a damn anymore what Marques Brownlee says about Tesla or Elon Musk. After the Cybercab event in October 2024 he said there was "just no way" Tesla would sell it for $30,000 before 2027, and promised to shave his head on camera if it did. Yesterday Tesla started takin" https://x.com/TheCaptainEli/status/2121502094762836023
- X @SaycheeseDGTL (1,686,150 followers, 2,565 likes, 2026-11-14): "Tesla begins offering its Cybercab for personal use to customers at $29,990, with a $1,000 deposit due at order &amp; a required $299/mo subscription after the first month. Delivery next month https://t.co/0Q39CuCBbE" https://x.com/SaycheeseDGTL/status/2121471755860709709
- Hacker News story "Tesla Opens Cybercab Orders to 'Select Customers' at $29,990 Plus $299 per Month" (28 points, 44 comments) https://news.ycombinator.com/item?id=50492178
- Hacker News story "Tesla Cybercab orders open at $29,990 and $299/mo, only in Robotaxi zones" (9 points, 0 comments) https://news.ycombinator.com/item?id=50496652
- YouTube "Elon JUST Started Selling Cybercabs for $299/mo" (Meet Kevin, 125,363 views) https://www.youtube.com/watch?v=pg1ucywxTh0
  - comment (88 likes): "If you miss the payment it’ll drive it self away 😂😂"
  - comment (63 likes): "I would buy the cybercab for my elderly parents in Nashville. Just knowing they don't have to drive is worth it for me."
- YouTube "Elon wants you to "buy" a Tesla Cybercab" (Chris Norlund, 92,264 views) https://www.youtube.com/watch?v=72YoKQBKEUw
  - comment (334 likes): "I will never give that rat one cent. Ever."
  - comment (282 likes): "Why would you pay 30 grand and then $300 a month just to get the finger from complete strangers everywhere you drive?"
- YouTube "Tesla Launches PERSONAL Cybercab Ownership Program ⚡️" (Electrified, 42,681 views) https://www.youtube.com/watch?v=G5dvZ6tHFb0

**What the judges wrote:**
- A1: "Tesla opening personal orders for a steering-wheel-free Cybercab at $29,990 plus $299 a month is a concrete consumer milestone for autonomy, alongside 24/7 robotaxi service in eight markets. The facts come through InsideEVs and Teslarati quoting Tesla's own post, and the number of invitees and the fleet terms are unknown. It touches something viewers could buy, and it has by far the highest engagement."
- A2: "Tesla's own Robotaxi X post, reported by several outlets, shows that a car with no steering wheel can be bought for personal use at $29,990 plus $299 a month, with December deliveries planned in Austin. The Tesla order page itself was not fetched, and the number of units is unstated. A consumer can now order a driverless vehicle, which makes this a concrete autonomy milestone, and it has the most viewer pull and momentum of the week."
- B1: "Several outlets quote Tesla's own announcement (no Tesla primary page fetched): $29,990 plus $299/month, a steering-wheel-free car offered for personal purchase, deliveries estimated December, initial availability limited and personal use only. It is the first consumer sale of a no-controls autonomous vehicle, a thing a viewer could buy, but it is limited and unconfirmed in volume. Biggest engagement of the week, with clear visuals and a price debate."
- B2: "Tesla offers the driverless Cybercab for personal purchase at $29,990 plus $299 a month, with December delivery in Austin, and 24/7 unsupervised robotaxi in eight cities. Sourced via trade press quoting Tesla, with limited availability and fine print. Something a viewer could actually buy, with by far the biggest public reaction."
- C1: "A car with no steering wheel offered for personal purchase at $29,990 plus a $299/month subscription, with deliveries estimated for December in Austin, is a real change in what a consumer can buy. The card's Tesla statements are all as quoted by InsideEVs and Teslarati (no Tesla page fetched), availability is 'select customers' in geofenced robotaxi zones with a federal 2,500-vehicle cap, and no unit counts are given, so the scale is unknown. Momentum is huge (25k likes, 282k YouTube views, HN 28) and the viewer could literally buy this."
- C2: "Tesla's Robotaxi account said select customers can order a Cybercab for personal use at $29,990 plus a $299/month autonomy subscription, with a $1,000 deposit and estimated December delivery in Austin, while unsupervised Robotaxi now runs 24/7 in eight cities; the sources are InsideEVs, Teslarati and Not a Tesla App quoting Tesla, and no Tesla page was fetched directly. How many invitations, when public sales open, and whether owners can join the fleet are all unknown, and the 2,500-per-year federal cap on noncompliant vehicles bounds it. A steering-wheel-free car offered to private buyers is a first the viewer could actually buy, and it was the most-shared card of the week (25,067 likes, 282k YouTube views); the video must state that these are Tesla's claims about an invite-only program."

**Visuals:** InsideEVs: lead photo of Tesla Cybercab at LA Auto Show 2024; second Cybercab photo; Not a Tesla App: two Cybercab photos; embedded Robotaxi X post with image; Teslarati: header photo of Austin robotaxi launch; embedded X posts (Robotaxi San Antonio announcement, geofence list, Cybercab subscription terms thread); Cybercab sale-page image

**Links:**
- https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/
- https://www.teslarati.com/tesla-cybercab-arrives-steep-ownership-fine-print/
- https://www.notateslaapp.com/news/4991/tesla-robotaxi-now-runs-247-in-every-city
- https://x.com/TheHumanoidHub/status/2120362083959140663
- https://x.com/clankrmedia/status/2121661050823049409
- https://www.straitstimes.com/business/tesla-invites-select-customers-to-buy-cybercab-robotaxi-for-personal-use
- https://www.forbes.com/sites/bradtempleton/2026/11/13/tesla-sales-of-self-driving-cybercab-coming-soon-to-invitees-20hr/
- https://electrek.co/2026/11/13/tesla-cybercab-personal-use-orders-299-subscription-geofence/
- https://news.google.com/rss/articles/CBMilwFBVV95cUxQVFA3STVYaFRsY056bXBKdkJmS1Q1LVFWZl80RUluY0syZWZHTlFWOUV3dl85V1k5R2FRSFdfcXdSc09ZMkNjOWZCNjl5eEdhMHo4T3hHeVF1anZCMlBzTDNaenExQXFuN1hIcEpyWnM0RzBpV1c5OUhxUHJZOVFoY3NqQjVlQ05qT0YyM1lTRWVFTTAxYl9v?oc=5
- https://www.inkl.com/news/tesla-opens-cybercab-orders-to-select-customers-at-29-990-plus-299-per-month
- https://www.newsbreak.com/insideevs-com-393662672/5006090249202-tesla-opens-self-driving-cybercab-orders-to-select-customers-at-29-990
- https://www.newsbreak.com/news/5006248168070-tesla-cybercab-orders-open-at-29-990-299-mo-only-in-robotaxi-zones
- https://economictimes.indiatimes.com/world/us/tesla-is-now-selling-its-self-driving-cybercab-robotaxi-for-personal-use-orders-open-for-29990-who-will-be-able-to-buy-and-where-it-will-work/articleshow/135823704.cms

### 3. Google DeepMind: Gemini Robotics 2.5; Apptronik Apollo 2 pilot at Mercedes-Benz

Status: **confirmed**. Primary: https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/

**Source quotes:**
- "Today, we are introducing Gemini Robotics 2.5, our most capable vision-language-action (VLA) model to date." (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)
- "We evaluated the model across 200 distinct tasks" (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)
- "Gemini Robotics 2.5 achieves a 30% to 100% higher success rate depending on the type of task." (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)
- "while completing tasks 20% to 30% faster" (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)
- "During a real-world proof-of-concept on Mercedes-Benz production lines," (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)
- "Achieving a 96% success rate, this pilot shows promising progress toward widespread real-world industrial applications." (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)
- "In our evaluations, this capability boosted safe task completion around obstacles by 33% and around nearby people by 43%." (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)
- "Argon achieves a 99.5% jailbreak resistance" (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)
- "a significant improvement over the 82% rate of our previous baseline" (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)
- "Gemini Robotics 2.5 is rolling out to our robotics partners." (https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)

**Not stated in the source:**
- Trial counts, success definition and variance not stated
- Comparisons are only against DeepMind's own prior Gemini Robotics models
- Public API, preview or trusted-tester availability not stated (only 'rolling out to our robotics partners')
- Model size, architecture and training data not stated
- Details of the 33% / 43% safety evaluations not stated
- Scale, duration and commercial terms of the Mercedes-Benz / Apptronik pilot not stated
- Full list of robot platforms used in the 200-task evaluation not stated

**What people say** (from the card):
- X @GoogleDeepMind (1,610,187 followers, 4,645 likes, 2026-11-10): "Gemini Robotics 2.5 is here. 🤖 Our latest physical AI is able to learn hundreds of everyday skills - from folding laundry and serving pastries to autonomously sorting car parts on the factory floor. Here’s a snapshot of its capabilities 🧵" https://x.com/GoogleDeepMind/status/2120169787896525104
- X @Google (31,895,787 followers, 1,911 likes, 2026-11-11): "Introducing Gemini Robotics 2.5, our most capable vision-language-action (VLA) model to date. 🤖 It increases both the breadth of skills a robot can learn and the speed at which it acquires them — helping robots step out of the lab and into real-world environments. Learn more abou" https://x.com/Google/status/2120549213826093271
- X @normandipalo (2,432 followers, 952 likes, 2026-11-10): "Today we announce Gemini Robotics 2.5, our new foundation model and the first that was strongly accelerated by Gemini 4 Argon. It is not a secret that we are pushing on recursive self-improvement (RSI), and in robotics it's paying off already. Gemini 4 Argon accelerated and impro" https://x.com/normandipalo/status/2120173936323244325
- X @GoogleDeepMind (1,610,187 followers, 375 likes, 2026-11-10): "Gemini 4 Argon helped build Gemini Robotics 2.5, speeding up months of training. It creates virtual practice spaces from a single camera view, and watches replays of its own mistakes to fix its technique, driving stronger performance on the real robot." https://x.com/GoogleDeepMind/status/2120169790853513261
- Hacker News story "Gemini Robotics 2.5" (6 points, 1 comments) https://news.ycombinator.com/item?id=50453617
- YouTube "Gemini Robotics 2.5 Reached 96% Success In A Mercedes Pilot" (AIM Network, 2,466 views) https://www.youtube.com/watch?v=qst0LXZ_6rg
- YouTube "Gemini Robotics 2.5, Clearly Explained" (Simply AI, 1,297 views) https://www.youtube.com/watch?v=UFIgJGi5GB4
- YouTube "Gemini Robotics 2.5: DeepMind's Robot Model, Read and Highlighted" (Standarity, 12 views) https://www.youtube.com/watch?v=1OOCY7LvX3o

**What the judges wrote:**
- A1: "Gemini Robotics 2.5 pairs a new VLA model (30-100% higher success across 200 tasks) with a real Mercedes-Benz production-line pilot at 96% success on Apollo 2 humanoids. Comparisons are only against DeepMind's earlier models, and the pilot's scale and duration aren't stated, so evidence is official but partly vague. Strong footage, a car brand viewers know, and top engagement."
- A2: "DeepMind's blog reports a new VLA model with 30-100% higher success across 200 tasks and a 96% success rate in a pilot on Mercedes-Benz production lines with Apollo 2 humanoids. The gains are measured only against DeepMind's own earlier models, and the pilot's scale is not stated. There is strong footage, and the momentum is high."
- B1: "Official DeepMind blog with specific numbers (200 tasks, 96% on a Mercedes line, 33%/43% safety gains) but comparisons are only against its own earlier models and trial counts are not stated. A new flagship VLA model plus a real factory pilot is a step the field notices. Many charts, demos and a photo to show, and Mercedes gives the viewer a known brand."
- B2: "DeepMind's primary post gives a new VLA model with a 200-task evaluation, 96% on a Mercedes-Benz production-line pilot with Apptronik humanoids, and many charts and demo videos. Gaps: comparisons only against its own prior models, no trial counts. A major lab moving toward real factory use is the week's strongest combination of significance and showability."
- C1: "An official DeepMind post with specific numbers (200 tasks, 30-100% higher success, 20-30% faster, 96% on a Mercedes-Benz line) plus embedded demo videos and a photo of two Apollo 2 humanoids sorting parts on a real automotive line. The comparisons are only against DeepMind's own prior model and trial counts, variance and pilot scale are missing, so the numbers are self-reported. Still, a frontier lab's new VLA running humanoids in a carmaker's factory is a concrete step, and it is the week's most-liked robotics post with 12 harvest items."
- C2: "DeepMind's primary post gives specific numbers (200 tasks, 30-100% higher success, 20-30% faster, 96% success in a Mercedes-Benz proof of concept with Apptronik Apollo 2) plus bar charts and six videos including humanoids sorting parts on a car line. The comparisons are only against DeepMind's own prior models and trial counts, success definitions and pilot scale are missing, so the video must flag that the figures are self-reported. The field's main accounts and Google itself pushed it (4,645 likes, 12 harvest items, 3.8k YouTube views), and a big lab's new model running in a real car plant is a step the whole field notices."

**Visuals:** Eight bar charts comparing Gemini Robotics 2 vs 2.5 (tabletop pick-and-place, multi-step/contact-rich, precision kitting and insertion, dexterous manipulation); Six embedded videos: hero video, two demo videos, Apollo 2 kitting clip, safety steerability clip, Gemini 4 Argon simulation clip; Photo: two Apollo 2 humanoids sorting parts on an automotive line; Hero image

**Links:**
- https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/
- https://x.com/carolina_parada/status/2120182324474388935
- https://x.com/GoogleDeepMind/status/2120169787896525104
- https://x.com/lukas_m_ziegler/status/2120173998982017211
- https://x.com/TheHumanoidHub/status/2119907053934399873
- https://x.com/TheHumanoidHub/status/2119904081946743130
- https://x.com/XRoboHub/status/2120830923503775978
- https://x.com/TheHumanoidHub/status/2120542217324360180
- https://x.com/clankrmedia/status/2120234085737980143
- https://x.com/Google/status/2120549213826093271
- https://x.com/TheHumanoidHub/status/2120658917688504781
- https://x.com/clankrmedia/status/2120067981245890711
- https://industrialnews.co.uk/google-tests-humanoid-robots-at-mercedes-benz-plants/

### 4. UC Berkeley: SuperMimic humanoid control paper (CoRL 2026)

Status: **unconfirmed**. Primary: https://arxiv.org/abs/2610.28480

**Source quotes:**
- "[Submitted on 20 Oct 2026 (v1), last revised 9 Nov 2026 (this version, v2)]" (https://arxiv.org/abs/2610.28480)
- "In 3 hours on a single GPU, SuperMimic surpasses the return of a state-of-the-art motion tracker baseline trained for 7 days on 128 GPUs." (https://arxiv.org/abs/2610.28480)
- "It reduces GPU-hours by more than $1000\times$ and fails on only 25 of 129,514 benchmark motions, versus 179 failures for the baseline. After 9.8 hours on two GPUs, SuperMimic achieves zero failures." (https://arxiv.org/abs/2610.28480)
- "Our policy transfers to a physical Unitree G1 across diverse and acrobatic motions." (https://arxiv.org/abs/2610.28480)
- "Accepted at CoRL 2026 (Oral)." (https://arxiv.org/abs/2610.28480)
- "Compute accounting compares SONIC’s reported 128 NVIDIA L40 GPUs for 7 days (about 21,000 GPU-hours) with 3 hours on one NVIDIA H100, within which SuperMimic surpasses the return of SONIC’s released checkpoint." (https://supermimic.github.io)
- "SuperMimic trains on 264.56 hours of BONES-SEED, the public subset of SONIC’s 611 hours: 2.3× less data." (https://supermimic.github.io)
- "Benchmark motions are training motions for both methods, so the counts measure coverage, not generalization." (https://supermimic.github.io)
- "Award winners will be announced at the Awards & Farewell session on Wednesday, November 11 (Bass Hall)." (https://www.corl.org/program/awards)
- "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model" (https://www.corl.org/program/awards)

**Not stated in the source:**
- CoRL awards page lists SuperMimic under 'Award Finalists' and does not list any award winners
- No source fetched states that SuperMimic won CoRL 2026 Best Paper
- arXiv v1 was submitted 20 Oct 2026 (before the window); v2 revised 9 Nov 2026
- No code repository listed on the project page
- Baseline compared is SONIC's released checkpoint; SONIC figures are as reported by SuperMimic authors

**What people say** (from the card):
- X @qiayuanliao (5,424 followers, 1,209 likes, 2026-11-12): "SuperMimic has won the CORL’s best paper award. Thanks for everyone’s support! https://t.co/8Ummvh55rm" https://x.com/qiayuanliao/status/2120740445923823846
- X @qiayuanliao (5,424 followers, 584 likes, 2026-11-10): "SuperMimic is open-sourced during my talk at CORL. Have fun with it: https://t.co/VX3Z0cZlk7 https://t.co/YaRp8dNBhm" https://x.com/qiayuanliao/status/2120288156742258824
- X @daibond_alpha (3,096 followers, 100 likes, 2026-11-10): "SuperMimic trains a humanoid tracker in 3 hours on a single GPU, surpassing a SoTA baseline trained for 7 days on 128 GPUs. What's the secret? Together with @fei_yang_wu, we dug in. Spoiler: it's differentiable planning with stable and effective gradients. 🧵 (1/5) Blog: https://t" https://x.com/daibond_alpha/status/2119977836002058462
- X @clankrmedia (5,535 followers, 9 likes, 2026-11-12): "A humanoid controller trained in 3 hours on 1 GPU. UC Berkeley's SuperMimic just won Best Paper at CoRL 2026. The previous state of the art took 128 GPUs for 7 days. One controller covers 129,514 motions, flips included. https://t.co/mgTIf6APBC" https://x.com/clankrmedia/status/2120872346076467507

**What the judges wrote:**
- A1: "SuperMimic matches a baseline that took 128 GPUs for 7 days in 3 hours on one GPU (more than 1000x fewer GPU-hours) and transfers to a real Unitree G1, a concrete research leap. It is accepted as a CoRL oral and is an award finalist, but the best-paper win comes only from the author's post, so the video should frame it as unconfirmed. Great visuals (flips, the 1-GPU vs 128-GPU photo) and strong engagement for a paper."
- A2: "The arXiv paper and project page report concrete numbers: 3 hours on one GPU beats a baseline that used about 21,000 GPU-hours, with real Unitree G1 transfer and an oral at CoRL. The best-paper win rests only on the author's post, since the awards page lists the paper as a finalist, so the video should say 'finalist, reportedly won'. The paper states that its benchmark measures coverage, not generalization. It is the strongest research result of the week, with excellent visuals."
- B1: "arXiv paper with specific numbers (3 hours on 1 GPU beats 7 days on 128 GPUs, 1000x fewer GPU-hours, real Unitree G1 clips) accepted as a CoRL oral. The Best Paper claim comes only from the author's own post while the CoRL page lists finalists, so I treat it as unconfirmed and do not rely on it. A concrete, explainable research result with strong engagement and plenty of video."
- B2: "Strong paper: trains a humanoid tracker in 3 hours on one GPU, beating a baseline that used 7 days on 128 GPUs, with real Unitree G1 transfer and CoRL oral acceptance. The Best Paper claim comes only from the author's posts; the CoRL page lists it as a finalist, so it stays unconfirmed. Excellent visuals and a clear 'compute collapse' story."
- C1: "A CoRL 2026 oral with a startling, specific claim: a humanoid whole-body controller trained in 3 hours on one GPU beats a baseline that took 7 days on 128 GPUs, covers 129,514 motions with 25 failures (zero after 9.8 hours), and transfers to a real Unitree G1 doing flips. The project page is rich in footage (flips grid, real-robot clips, one-GPU-vs-128 photo) and the compute comparison is easy to explain. The 'won Best Paper' claim comes only from the author's post; the CoRL page lists it as a finalist, so the video must say so. Baseline figures are as reported by the authors."
- C2: "The arXiv paper and project page give concrete numbers: a humanoid whole-body controller trained in 3 hours on one H100 that surpasses a baseline trained 7 days on 128 GPUs, 25 failures out of 129,514 motions versus 179, zero failures after 9.8 hours, and transfer to a real Unitree G1 doing flips, with a CoRL 2026 Oral. The best-paper claim comes only from the author's post (1,209 likes); the CoRL awards page lists it as a finalist and no fetched source confirms a win, so the video should say 'oral and award finalist, authors say it won'. Baseline figures are as reported by the authors and the benchmark measures coverage not generalization, but the footage, the one-GPU-versus-GPU-wall photo and the compute chart make it the week's most showable research result."

**Visuals:** Project page: hero montage video; full film (1080p mp4); compute chart video and interactive return-vs-GPU-hours chart; photo of one GPU in front of a 128-GPU wall; flips grid video; real Unitree G1 clips; simulator vs world-model comparison videos (policy off, policy on, slippery floor); in-browser interactive demo; alphaXiv: author X thread with video; arXiv: abstract page with PDF

**Links:**
- https://arxiv.org/abs/2610.28480
- https://supermimic.github.io
- https://www.corl.org/program/awards
- https://x.com/clankrmedia/status/2120872346076467507

### 5. Anthropic: Claude Fable 5.2 robot-control results

Status: **confirmed**. Primary: https://www.anthropic.com/claude-fable-and-mythos-5-2

**Source quotes:**
- "Compared with earlier Claude models, Fable 5.2 shows substantially improved skills in controlling physical robots" (https://www.anthropic.com/claude-fable-and-mythos-5-2)
- "The video below shows robots controlled by Fable 5.2 completing a variety of tasks" (https://www.anthropic.com/claude-fable-and-mythos-5-2)
- "solving a puzzle cube, unscrewing a bottle cap, and drawing a picture of a robot" (https://www.anthropic.com/claude-fable-and-mythos-5-2)
- "On the public RoboDojo benchmark, which tests how AIs can control a simulated two-arm robot from camera images" (https://www.anthropic.com/claude-fable-and-mythos-5-2)
- "Fable 5.2 outperformed other frontier models that were run on the same harness" (https://www.anthropic.com/claude-fable-and-mythos-5-2)
- "as well as most AI systems built specifically to control robots." (https://www.anthropic.com/claude-fable-and-mythos-5-2)
- "At Max effort, Fable 5.2 scores 64.9%, more than double GPT-6 Astra (29.8%) and Fable 5.1 (27.6%)." (https://www.anthropic.com/claude-fable-and-mythos-5-2)
- "Scores are based on our internal runs for each model using the highest available effort level." (https://www.anthropic.com/claude-fable-and-mythos-5-2)
- "RoboDojo Sim shows the highest-available effort and was run internally." (https://www.anthropic.com/claude-fable-and-mythos-5-2)
- "RoboDojo Sim tests control of a simulated two-arm robot." (https://www.anthropic.com/claude-fable-and-mythos-5-2)

**Not stated in the source:**
- Number of tasks, episodes or trials per model not stated
- Variance or confidence intervals not stated
- Success criteria / metric definition not stated
- Harness details (observation format, action space, prompts, time limits) not described
- Whether the video shows simulated or physical robots not stated; robot hardware not named
- No robotics score given for Claude Mythos 5.2
- The 'AI systems built specifically to control robots' are not named and no scores are given for them
- Benchmarks table gives no RoboDojo score for Gemini 4 Carbon

**What people say** (from the card):
- X @AndrewCurran_ (117,345 followers, 2,131 likes, 2026-11-13): "Anthropic reported a similar sudden and unusual jump in robotics capabilities with Fable 5.2. As the general intelligence of frontier models scales up, it is increasingly surpassing specialized training. This will ultimately generalize to every sphere of human ability. https://t." https://x.com/AndrewCurran_/status/2121337915087433772
- X @emollick (418,943 followers, 1,524 likes, 2026-11-10): "Looks like Fable 5.2 is above the already steep trendline on three of the most informative unsaturated benchmarks (ECI is a composite of other benchmarks, Remote Labor measures the ability to do end-to-end freelance jobs, RoboDojo involves controlling robotics). https://t.co/LUBz" https://x.com/emollick/status/2120000888974766501
- X @DeryaTR_ (345,212 followers, 1,099 likes, 2026-11-09): "This is a very big deal! Fable 5.2 appears to have achieved physical general intelligence, the equivalent of AGI for robotics! I was not expecting this until later next year, and now it is clear 2027 will be the year of intelligent robots!" https://x.com/DeryaTR_/status/2119909440438591672
- X @kimmonismus (159,053 followers, 658 likes, 2026-11-10): "While we’re all focused on Claude Fable 5.2’s coding abilities, its work in science and engineering has received far less attention. Anthropic has made insane steps and solved crazy hard problems that many missed: - Hardware: Fable 5.2 designed three electronic devices from short" https://x.com/kimmonismus/status/2120103171414806648

**What the judges wrote:**
- A1: "A general-purpose frontier model scoring 64.9% on RoboDojo, more than double GPT-6 Astra and its own predecessor, is a big signal that general models are overtaking specialized robot controllers. The numbers are Anthropic's own internal runs, with no trial counts, metric definition or harness details, so evidence is an official but self-reported claim. The company video of real tasks (puzzle cube, bottle cap) and the bar chart make it easy to show, and leading AI accounts shared it widely."
- A2: "Anthropic's own page shows a large jump on RoboDojo (64.9% vs 29.8% for GPT-6 Astra), plus a video of real-world tasks. The numbers come only from internal runs, with no trial counts, variance or harness details, so the evidence is self-reported. Still, a general frontier model more than doubling rivals at robot control is a meaningful signal, and it drew wide attention."
- B1: "Official Anthropic post with specific RoboDojo numbers (64.9% vs 29.8%), but they are internal runs in simulation with no trial counts or harness details, and the video's sim/real status is unstated. Still, a general LLM beating most purpose-built robot controllers is a field-level shift. Widely discussed, though some X hype ('AGI for robotics') is one person's view."
- B2: "Anthropic's own post claims a jump to 64.9% on RoboDojo, a simulated two-arm benchmark, run internally with no trial counts, harness or variance, and it is unclear if the video is real hardware. If it holds, a general model beating robot-specific systems is significant, and the field is talking about it. Self-reported numbers cap the evidence."
- C1: "Anthropic's launch page reports Fable 5.2 scoring 64.9% on the public RoboDojo simulated two-arm benchmark, more than double GPT-6 Astra (29.8%) and its own predecessor (27.6%), with a video of robots solving a puzzle cube and unscrewing a bottle cap. All runs are internal, trial counts and metric definitions are absent, the 'specialized robot systems' it claims to beat are unnamed, and whether the video is real hardware is not stated. Together with OpenAI folding its robotics team (s02) and the Embodied First Exam benchmark (s04), it is the week's clearest evidence of a trend the field is debating: general models starting to drive robots."
- C2: "Anthropic's own launch page reports Fable 5.2 scoring 64.9% on the RoboDojo simulated two-arm benchmark against 29.8% for GPT-6 Astra and 27.6% for Fable 5.1, with a video of robots solving a cube and unscrewing a cap. Every number is from internal runs with no task counts, variance, harness details or named robot-specific baselines, and the video does not say whether the robots are real, so it is a vendor claim with a chart. It spread widely (2,131 and 1,524 likes from large accounts) and pairs with the OpenAI reorg as a trend, but a company praising its own product is not evidence and the significance depends on claims the card cannot confirm."

**Visuals:** Video: robots controlled by Fable 5.2 solving a puzzle cube, unscrewing a bottle cap, drawing a picture of a robot; Bar chart: 'RoboDojo (average)' max score by model (Fable 5.2 64.9, Opus 5.5 39.4, Gemini 4 Argon 34.2, GPT-6 Astra 29.8, Fable 5.1 27.6); Benchmarks table with RoboDojo Sim row

**Links:**
- https://www.anthropic.com/claude-fable-and-mythos-5-2
- https://x.com/clankrmedia/status/2120144768051167369

## Reserves

- 6. NVIDIA: CEO Jensen Huang resigns: panel 5.67, pairwise wins 1
- 7. Boston Dynamics: Atlas on wheels research test: panel 6, pairwise wins 0
- 8. Runway: Praxis-1 open weights: panel 5.67

## Not picked (judged)

- 9. OpenAI: robotics team reorganised into research and applied groups (5.0)
- 10. Agile Robots: $500M Series F (4.83)
- 11. Ant Group (Robbyant): LingBot-Touch tactile model (4.33)
- 12. ASUS: Gallop humanoid (3.83)
- 13. Path Robotics: $300M Series E (3.83)
- 14. GXO and Apptronik: letter of intent for up to 1,000 Apollo humanoids (3.67)
- 15. El Salvador and Haiti: armed robot dogs used in stadium operation (3.67)
- 16. Smart Analytics Global: Q3 2026 humanoid shipment estimate (3.67)
- 17. Washington, D.C.: council committee vote on robotaxi bill (3.5)
- 18. Embodied First Exam and LEASH: benchmarks of AI agents controlling robots (3.33)
- 19. Beijing, South Korea, Japan: government physical-AI plans (2.83)
- 20. India: IT ministry consultation on in-home worker video for physical-AI training (2.83)
- 21. Foundation (military humanoid startup): missed payroll; IDF sale talks (2.67)

## Calibration

Uncalibrated: the user's blind ranking for this week isn't recorded (`judge/calibration.json`).
