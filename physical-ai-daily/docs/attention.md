# Holding attention: what the evidence says (2026-11-14)

**Question from the user:** how do we keep viewers watching longer, and which background elements hurt attention?
Research before design: each rule below cites its evidence and states how strong it is.

## Findings

| # | Finding | Evidence | Strength |
|---|---|---|---|
| 1 | **Decorative extras hurt.** Interesting but irrelevant material ("seductive details") lowers learning. The effect is larger for illustrations than for text, and worse when the extra stays on screen constantly | Rey 2012 meta-analysis ([ResearchGate](https://www.researchgate.net/publication/257690772_A_review_of_research_and_a_meta-analysis_of_the_seductive_detail_effect)): illustrations d≈0.95 retention; Sundararajan & Adesope 2020 ([WSU summary](https://news.wsu.edu/2020/03/19/seductive-details-inhibit-learning/)): g≈−0.37 over 58 studies; Mayer's coherence tests ([summary PDF](https://edtechuvic.ca/wp-content/uploads/sites/11/2022/09/principles-for-reducing-extraneous-processing-in-multimedia-learning-coherence-signaling-redundancy-spatial-contiguity-and-temporal-contiguity-principles.pdf)): 23 of 23, median d≈0.86 | Strong (several meta-analyses). Estimates range from small to large, and one 2026 review reports g≈−0.16 |
| 2 | **New motion grabs the eye.** Motion onset captures attention even when it carries no information, and jerky motion captures it more | Abrams & Christ 2003 ([Springer](https://link.springer.com/article/10.3758/BF03193623)); jerkiness confound and debate ([Springer 2018](https://link.springer.com/article/10.3758%2Fs13414-018-1548-1)) | Strong lab evidence; the setting is basic attention research, not YouTube |
| 3 | **Sound over speech hurts recall.** Background music and sounds added to narrated lessons lowered learning | Moreno & Mayer 2000 ([PDF](https://tecfa.unige.ch/tecfa/teaching/methodo/Moreno_Mayer00.pdf)); one review argues placement matters more than avoidance: harm comes when music overlaps speech ([review](https://culturaesaude.med.br/en/background-music-increases-viewer-engagement-a-systematic-review-and-meta-analysis/)) | Strong for overlapping speech; the second source's venue is unverified |
| 4 | **Cues synced to the narration help.** Dynamic highlights that appear as the narrator mentions them pull gaze to the right place. Stacked visual + verbal cues, and arrows, can distract | Rodemer 2022 eye tracking ([Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/acp.3973)); 360° cue study ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11061496/)); dual-modality cues hinder ([ResearchGate](https://www.researchgate.net/publication/394212594_The_Dual-Modality_Cues_Hinder_Learning_in_Instructional_Videos_Evidence_from_an_Eye-Tracking_Study)); arrows distracted ([ACS](https://pubs.acs.org/jceda8/article/103/7/3599/5172104/Eye-Tracking-Informed-Design-of-Instructional)) | Moderate (small studies, consistent direction) |
| 5 | **Scene changes lift attention; visual clutter lowers it.** Cuts raise attentional synchrony about 0.7 s later, and visual complexity has a delayed negative effect | J. Academy of Marketing Science 2026, 2,520 viewing sessions ([Springer](https://link.springer.com/article/10.1007/s11747-025-01137-x)); average shot length shapes gaze ([ResearchGate](https://www.researchgate.net/publication/342285366_How_filmmakers_guide_the_eye_The_effect_of_average_shot_length_on_intersubject_attentional_synchrony)) | Moderate-strong (large sample, ads not explainers) |
| 6 | **Shorter videos hold people.** Engagement fell with length; the recommendation was chunks under 6 minutes. Enthusiasm and speaking rate matter | Guo, Kim & Rubin 2014, 6.9M edX sessions ([PDF](https://www.cs.rochester.edu/hci/pubs/pdfs/edX-MOOC-video-production-and-engagement_LAS-2014.pdf)) | Strong (huge observational sample); MOOC lectures, not YouTube |
| 7 | **A visible presenter reduces mind wandering** | Eye-tracking study 2023 ([T&F](https://www.tandfonline.com/doi/abs/10.1080/14703297.2023.2251955)) | Moderate; doesn't fit a faceless channel today |
| 8 | **Dark text on light reads better** for people with normal vision (positive-polarity advantage, via display luminance) | Piepenbrock et al. 2014 ([dissertation](https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-31413)); [NN/g summary](https://www.nngroup.com/articles/dark-mode/) | Strong for reading text; little evidence about video watch time |

**Not found:** a peer-reviewed study of YouTube's first-30-seconds drop-off. The figures that circulate ("50–70% leave
in 30 s") come from vendor blogs without a source, so we don't use them.

## What we changed (Blueprint design, v7)

| Rule | Why | Change |
|---|---|---|
| Static background | 1, 2: constant background motion is a seductive detail plus a motion cue that competes with the content | Blueprint grid no longer drifts. The faint pixel drift stays only in the unused candidate designs |
| No sound under the voice | 3 | Removed the section "tick", which overlapped the first words of a beat. The sonic logo plays only in silent gaps (sting, end tail) |
| Visual cues, synced and single | 4 | Section chips stay (visual signal timed to the beat); word-cued reveals and the highlighted source sentence stay. No arrows pointing at things, and no stacked cues |
| Simple scenes that change often | 5 | Keep one focal element per beat and a new visual every 5–7 s (already the format rule) |
| Keep episodes short | 6 | About 4 min per episode, under the 6-minute mark |
| Light version available | 8 | `"design": "blueprint-light"`: same layout on warm paper with ink text. It also avoids flashing between a dark frame and the mostly white source screenshots |

## Open questions (not yet tested)

- **Full verbatim captions alongside narration:** Mayer's redundancy principle says identical on-screen text can
  overload viewers, but many YouTube viewers watch muted. Keep the captions for now; this needs its own research and A/B
  test.
- **On-screen presenter (7):** would mean a face or an avatar; out of scope for a faceless channel.
- **Validation:** check YouTube Studio's retention curves once episodes are live. That data outranks any lab result for
  this channel.
