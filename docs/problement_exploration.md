# Problem Exploration

## Core Problem

The main issue is low user engagement and retention after the first couple of weeks. Users can be grouped into three types:

-**Type A:** Download the app and stick with it — they find value and keep coming back.

-**Type B:** Download the app but stop using it after a few weeks — they drop off quickly.

-**Type C:** New users — we don’t know yet whether they’ll stay or churn.

The goal is to help Type C users become more like Type A, and not like Type B. This means:

- Attracting users who are likely to engage long-term.
- Designing an early experience that gets them hooked and delivers real value from day one.

**Note:** While this case focuses on the app experience, marketing efforts are just as important. Targeting the right users from the start can make a big difference.

## Assumptions

- Users download the app because they have a wellness goal (fitness, stress relief, etc.).
- We can collect basic info during onboarding (goals, preferences).
- User behavior can be tracked (content interaction, app open frequency).
- Push notifications can be sent to users.

## Signals for Personalization

To power content matching, clustering, churn prediction, and daily planning, we focus on a few key early signals:

* **User-entered goal** : Captured at onboarding and used for semantic content matching.
* **Streak length** : Indicates consistency and potential habit formation.
* **App open patterns** : Frequency and time of day help infer routine vs. sporadic use.
* **Content preferences** : Types of content engaged with (e.g., meditations, workouts, challenges) and preferences.
* **Notification response** : Measures responsiveness to messages and prompts.
* **Daily plan follow-through** : Tracks whether users complete suggested plans.
