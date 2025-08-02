# Problem Statement and Proposal

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

## Proposal

### Personalization System for Early Retention

To enhance early user retention, this system personalizes the initial user experience by combining goal-based content matching, behavioral clustering, churn prediction, and a conversational micro-coach. Each component works together to adapt the app to the user's needs, making it feel supportive and relevant from the start.

### How It Works (Overview of Components)

1. **Goal-to-Content Matching**
   * At onboarding, users set a wellness goal in their own words.
   * An embedding model matches that goal to semantically relevant content in the catalog.
   * This ensures their very first experience feels aligned with their intent.
2. **Early Behavior Clustering**
   * After a few sessions (e.g., 3–5), user behavior is clustered based on engagement style: streaks, content preferences, time of day, etc.
   * Users are grouped into behavioral personas (e.g., “explorers,” “routine-builders,” “casual check-ins”).
   * The app adapts layout, tone, and notifications based on the cluster.
3. **Drop-Off Risk Predictor**
   * A simple binary classifier predicts churn risk after a few days of use.
   * Inputs include streak length, app open frequency, notification response, and content depth.
   * When risk is high, the app triggers a personalized intervention (e.g., a motivational message, reward, or new challenge).
4. **Micro-Coach with Daily Planning**
   * A conversational assistant checks in each day to help users set a small wellness plan (e.g., one meditation + one stretch).
   * It tailors suggestions based on the user’s goal, past activity, and engagement type.
   * The micro-coach also follows up, reinforcing habits and offering encouragement (Not done).

### How This Fits Into the User Journey

* **Day 0 (Signup)** : User selects a goal → app immediately recommends semantically matched content.
* **Days 1–5** : Behavioral data is collected → user is clustered → app adapts recommendations:
  * **Casual Check-ins**:
    * Encourage sporadic users with gentle reminders and motivational messages.
    * Highlight quick, low-effort activities to maintain interest.
  * **Routine Builders**:
    * Provide structured plans with clear milestones and progress tracking.
    * Suggest activities that align with their preference for routine.
  * **Explorers**:
    * Recommend diverse activities to keep engagement fresh and exciting.
    * Offer challenges or experiments to encourage consistent participation.
* **Day 3+** : Churn prediction model monitors usage → if risk is high, app intervenes with a message.
* **Daily** : Micro-coach checks in, builds a mini plan with the user.

### A/B Testing and user feedback

A/B testing and user feedback are crucial to evaluate the effectiveness of each component in the system. By comparing user engagement, retention rates, and satisfaction across different variations, we can identify what works best and refine the approach. This iterative process ensures the system continuously adapts to meet user needs and drive optimal outcomes.

### Additional idea

An end to end micro coach. It's explained in [Discovery Sprint](discovery_sprint.md).
