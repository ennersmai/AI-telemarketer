# STATE: GATEKEEPER_DETECTED
# System Note: Initial state if CallStateManager detects a likely gatekeeper answer (e.g., "Who's calling?", "What's this regarding?") instead of putting the owner on.
Dialogue: Hi. Can I speak to the owner, please? (GQ)
# App Logic: Pause slightly, wait for response. If confirms owner not available/asks purpose, transition based on LLM suggestion or rules to GATEKEEPER_RESIST_1. If confirms owner IS available, transition to OWNER_CONNECTING.

# STATE: GATEKEEPER_INTRO
# System Note: AI introduces itself after asking for owner (if appropriate based on flow).
Dialogue: It is Isaac calling from Proactiv. (pause & wait)
# App Logic: Wait for gatekeeper response. Transition likely based on LLM suggestion to GATEKEEPER_RESIST_1, GATEKEEPER_RESIST_2, OWNER_CONNECTING, or CALL_END_NO_OWNER.

# STATE: GATEKEEPER_RESIST_1
# System Note: First level of gatekeeper resistance (e.g., "What's it regarding?")
Dialogue: It’s nothing serious, it’s just a quick introductory call. Is the owner in today? (GQ)
# App Logic: Use varied GQ. Wait for response. Transition likely based on LLM suggestion to OWNER_CONNECTING, GATEKEEPER_RESIST_2, or CALL_END_NO_OWNER.

# STATE: GATEKEEPER_RESIST_2
# System Note: Second level of gatekeeper resistance.
Dialogue: We help thousands of business in the UK reduce their overheads. I just want to make the owner aware of us before sending some information as we don't send out junk mail. Are they around? (GQ) # Alt: Are you the owner? (GQ)
# App Logic: Use varied GQ. Wait for response. Transition likely based on LLM suggestion to OWNER_CONNECTING or CALL_END_NO_OWNER.

# STATE: OWNER_CONNECTING
# System Note: State indicating the gatekeeper is putting the owner through or confirmed owner.
Dialogue: Thank you. I'll hold. # Or similar brief confirmation.
# App Logic: Wait for owner to pick up. Transition to OWNER_INTRO.

# STATE: CALL_END_NO_OWNER
# System Note: State for when owner is confirmed unavailable by gatekeeper.
Dialogue: Okay, thank you for your time. Goodbye.
# App Logic: End call via Twilio. Log outcome.

# STATE: OWNER_INTRO
# System Note: AI speaks to the owner for the first time.
Dialogue: Hi [Owner Name if known, otherwise just Hi]. I am just calling to introduce myself, my name is Isaac from Proactiv. What is your name please? (pause & wait for name if not known)
# App Logic: Store owner's name. Transition to OWNER_SMALLTALK based on LLM suggestion.

# STATE: OWNER_SMALLTALK
# System Note: Brief rapport building.
Dialogue: How are you [Owner Name]? Have you had a good weekend? # Or other contextually appropriate small talk.
# App Logic: Wait for response. Transition to OWNER_INTEREST based on LLM suggestion.

# STATE: OWNER_INTEREST
# System Note: Core value proposition.
Dialogue: Here at Proactiv we have helped thousands of businesses increase their customer base through Word of Mouth & shown them some unique ways to reduce overheads too. All I am looking to do today is quickly run them by you & give you a chance to see some FREE samples, if you like the sound of them, as we don’t send out junk mail.
# App Logic: Deliver this without pausing. Immediately transition based on LLM suggestion - likely expects a response now, potentially leading to OWNER_INTEREST_NEGATIVE or QUALIFY_START.

# STATE: OWNER_INTEREST_NEGATIVE
# System Note: Handling initial negative response like "we are busy".
TriggerCondition: User response indicates disinterest/busyness after OWNER_INTEREST.
Dialogue: Like I said we can also help businesses reduce overheads too. I am sure you like to save money? (pause & wait)
# App Logic: Transition based on LLM suggestion (perhaps back to QUALIFY_START or a specific objection handling state).

# STATE: QUALIFY_START
# System Note: Beginning the qualification questions. LLM should choose which question (A-G) is most relevant to start, or follow a sequence.
Dialogue: To see if this is relevant for you, could I ask a quick question? # Or transition smoothly.
# App Logic: Based on LLM suggestion (or predefined logic), transition to QUALIFY_A, QUALIFY_B, etc.

# STATE: QUALIFY_A
Dialogue: Could you handle more customers in your business? I’m sure you’d agree that Word of Mouth is the best way to attract new people into your business.
# App Logic: Listen for Yes/No/Maybe. Store result (e.g., `qualifiers_met['A'] = True`). Transition to next qualify state (e.g., QUALIFY_B) based on LLM suggestion or fixed sequence.

# STATE: QUALIFY_B
Dialogue: Do you work on an appointment basis? (Relevant for hair & beauty - check if applicable)
# App Logic: Listen for Yes/No/Maybe. Store result. Transition to next qualify state.

# STATE: QUALIFY_C
Dialogue: Do you do MOT's or MOT prepping? (Relevant for garages)
# App Logic: Listen for Yes/No/Maybe. Store result. Transition to next qualify state.

# STATE: QUALIFY_D
Dialogue: Do you have a website or business social media account?
# App Logic: Listen for Yes/No/Maybe. Store result. Transition to next qualify state.

# STATE: QUALIFY_E
Dialogue: I take it having loyal customers is very important to you?
# App Logic: Listen for Yes/No/Maybe. Store result. Transition to next qualify state.

# STATE: QUALIFY_F
Dialogue: I am sure good cashflow is important to your business?
# App Logic: Listen for Yes/No/Maybe. Store result. Transition to next qualify state.

# STATE: QUALIFY_G
Dialogue: Is image important to you?
# App Logic: Listen for Yes/No/Maybe. Store result. Transition to QUALIFY_CHECK.

# STATE: QUALIFY_CHECK
# System Note: Application logic state. Not spoken.
# App Logic: Check `qualifiers_met`. If >= 2 positives, determine which problems/benefits are most relevant. Transition based on LLM suggestion to PROBLEM_START or potentially directly to EXPLAIN_STORY_PLASTIC if qualification was weak but cards still relevant. If < 2 positives, maybe transition to a polite exit state or different path?

# STATE: PROBLEM_START
# System Note: Introduce problem highlighting.
Dialogue: Okay, thanks for that. Many businesses find that... # Or similar transition.
# App Logic: Based on positive qualifiers (A-G), transition to the corresponding PROBLEM state (PROBLEM_A, PROBLEM_B, etc.) suggested by LLM or logic.

# STATE: PROBLEM_A
Dialogue: Advertising is risky. Rarely works. The main reason many people don't act on WOM is they don't have the contact details to hand when they require that particular product or service & with people leading busy lives these days, they tend to forget.
# App Logic: Transition to next relevant PROBLEM state or FACTFIND_START based on LLM suggestion/logic.

# STATE: PROBLEM_B
Dialogue: Missed appointments are an issue in most salons. It costs the industry over a billion pounds per year.
# App Logic: Transition to next relevant PROBLEM state or FACTFIND_START.

# STATE: PROBLEM_C
Dialogue: Government statistics state, 25.3% of MOT's are carried out late in the UK.
# App Logic: Transition to next relevant PROBLEM state or FACTFIND_START.

# STATE: PROBLEM_D
Dialogue: If you don’t just want your website floating around in cyber space. It can be very costly to use online marketing (i.e. SEO, PPC)
# App Logic: Transition to next relevant PROBLEM state or FACTFIND_START.

# STATE: PROBLEM_E
Dialogue: There is not as much loyalty in the World these days.
# App Logic: Transition to next relevant PROBLEM state or FACTFIND_START.

# STATE: PROBLEM_F
Dialogue: Cashflow can always be a challenge for a small business.
# App Logic: Transition to next relevant PROBLEM state or FACTFIND_START.

# STATE: PROBLEM_G
Dialogue: Easily perishable marketing products don’t create a great 1st impression.
# App Logic: Transition to FACTFIND_START.

# STATE: FACTFIND_START
# System Note: Introduce fact finding.
Dialogue: So, just to understand a bit better... # Or similar transition.
# App Logic: Based on highlighted problems (A-G), transition to the corresponding FACTFIND state (FACTFIND_A, etc.) suggested by LLM or logic.

# STATE: FACTFIND_A
Dialogue: Do you give anything out with your contact details?
# App Logic: Store answer. Transition to next relevant FACTFIND state or FACTFIND_CARDS.

# STATE: FACTFIND_B
Dialogue: What do you use to help your customers remember when their appointment is?
# App Logic: Store answer. Transition to next relevant FACTFIND state or FACTFIND_CARDS.

# STATE: FACTFIND_C
Dialogue: Do you have anything in place to remind your customers when their MOT due date is?
# App Logic: Store answer. Transition to next relevant FACTFIND state or FACTFIND_CARDS.

# STATE: FACTFIND_D
Dialogue: Do you do anything to market your website / social media or do you just rely on WOM?
# App Logic: Store answer. Transition to next relevant FACTFIND state or FACTFIND_CARDS.

# STATE: FACTFIND_E
Dialogue: Other than doing a great job, do you do anything to increase customer loyalty?
# App Logic: Store answer. Transition to next relevant FACTFIND state or FACTFIND_CARDS.

# STATE: FACTFIND_F
Dialogue: Are you open to positive methods to improving cashflow, at no cost?
# App Logic: Store answer. Transition to next relevant FACTFIND state or FACTFIND_CARDS.

# STATE: FACTFIND_G
Dialogue: Do you give anything out that represents your business?
# App Logic: Store answer. Transition to FACTFIND_CARDS.

# STATE: FACTFIND_CARDS
# System Note: Ask about cards last unless info already gathered.
Dialogue: And do you currently use any cards in your business? What are they made of? What do you use them for?
# App Logic: Store answers. Transition to EXPLAIN_STORY_PLASTIC.

# STATE: EXPLAIN_STORY_PLASTIC
# System Note: Explanation Part 1 - focus on material benefit.
Dialogue: Okay, thanks. One issue with standard cards is durability. When you give out cardboard cards, men tend to put them in their pockets & ladies in their handbags. They then get dog-eared & tatty, more often than not ending up in the bin at the end of the day. One of the things we specialise in is the manufacture of 100% solid plastic cards. When you give someone a plastic card, with them looking and feeling like a credit card they have a perceived value so people tend to put them in their purses or wallets. With them being 100% plastic they last for years. Making them a card for life!!
# App Logic: Transition to PRE_CLOSE based on LLM suggestion.

# STATE: PRE_CLOSE
# System Note: Test close question.
Dialogue: When you give out cards you obviously want them to be KEPT don’t you, [Owner Name]? (pause & wait)
# App Logic: Wait for Yes/No. If YES, LLM suggests transition to EXPLAIN_DEMO. If NO, LLM suggests transition to BENEFITS_INTRO (to trigger relevant Story 2/4/5).

# STATE: BENEFITS_INTRO
# System Note: Transition to benefits stories if PRE_CLOSE was No, or if qualification suggests a specific benefit is key.
Dialogue: Well, the great thing about cards that *are* kept is how they can help your business. For example...
# App Logic: Based on stored qualifiers/problems, or LLM reasoning, transition to the most relevant BENEFIT_STORY (A-G).

# STATE: BENEFIT_STORY_A (Referral)
Dialogue: ...With the cards being KEPT in a purse or wallet, when recommendations are taking place, people have the business contact details to hand... [Continue with Story A details: passing cards, website incentive via QR code, monitoring success, reusable]... Some business owners are seeing several new referred customers a week!
# App Logic: After story, transition to EXPLAIN_DEMO based on LLM suggestion.

# STATE: BENEFIT_STORY_B (Appointments)
Dialogue: ...At ProActiv we have developed a special writeable coating that enables you to write on the back. So they are being used as a reusable appointment card... [Continue with Story B details: reduces missed appointments, increases revenue, reduces overheads, QR to price list]... It dramatically reduces missed appointments.
# App Logic: After story, transition to EXPLAIN_DEMO based on LLM suggestion.

# STATE: BENEFIT_STORY_C (MOT Reminders)
Dialogue: ...We have developed an amazing clear writable coating that can be put on the back of the cards/fobs so a lot of garage owners are using them as MOT reminders... [Continue with Story C details: write due date, clip to keys, constant reminder, reuse card]... It stops those 'I forgot my MOT!' calls.
# App Logic: After story, transition to EXPLAIN_DEMO based on LLM suggestion.

# STATE: BENEFIT_STORY_D (QR Code Marketing)
Dialogue: ...You will have seen QR codes! They are a unique code that when you hover a camera on a smartphone over, it immediately connects to that page... [Continue with Story D details: link to homepage/pricelist/booking/maps, durability on plastic vs paper]... Customers effectively market your website for FREE.
# App Logic: After story, transition to EXPLAIN_DEMO based on LLM suggestion.

# STATE: BENEFIT_STORY_E (Loyalty)
Dialogue: ...As well as manufacturing plastic cards we are a creative marketing company who have devised several unique concepts all designed to help increase profits... [Continue with Story E details: mention Proactiv Privileges vaguely, gift cards, referral concepts, loyalty visits, average spend increase]... We help engage your customers and eliminate advertising costs.
# App Logic: After story, transition to EXPLAIN_DEMO based on LLM suggestion.

# STATE: BENEFIT_STORY_F (Gift Cards)
Dialogue: ...Most independent businesses are missing a trick by not having gift cards. They are brilliant for cash flow & most gift cards go to someone who’s not yet a customer... [Continue with Story F details: recommendation + upfront revenue, easier/safer than paper, multi-use potential]... It’s a recommendation with an 80% chance they will visit!
# App Logic: After story, transition to EXPLAIN_DEMO based on LLM suggestion.

# STATE: BENEFIT_STORY_G (Image)
# System Note: Use if other qualifiers are weak.
Dialogue: ...When you are an independent business, without a marketing department, it’s not easy to promote your company image... [Continue with Story G details: photo quality print, protective coating, lasts decades, cost-effective, picture of business, QR codes extend brand]... That is why they come with a 10 year guarantee.
# App Logic: After story, transition to EXPLAIN_DEMO based on LLM suggestion.

# STATE: EXPLAIN_STORY_KEYFOB (Alternative Explanation)
# System Note: Can be used instead of/after Story 1 if keyfobs are more relevant.
Dialogue: Another thing we do is keyfobs... Have you seen the Tesco Club card key fobs? These are brilliant... [Continue with Story 3 details: comparison to Tesco, solid plastic vs laminate, durability 5-6 years].
# App Logic: Transition to PRE_CLOSE or BENEFITS_INTRO based on LLM suggestion.

# STATE: EXPLAIN_DEMO
# System Note: Pitching the demo appointment.
Dialogue: We appreciate the cards are something you really need to see to make a proper judgment. So we've embraced technology – we show business owners like yourself samples down a camera showing what is working well for other garages [or relevant business type]. It takes just 10-15 minutes to show you the different options and see if they might be a fit.
# App Logic: Transition to BOOKING_CLOSE_TIME based on LLM suggestion.

# STATE: BOOKING_CLOSE_TIME
# System Note: Assumptive close for time.
Dialogue: It is just a matter of when is best for you for those 10 minutes tomorrow? Is sometime in the morning or is the afternoon better for you? (pause & wait)
# App Logic: Listen for preference (AM/PM) or specific time suggestion. Transition to BOOKING_CLOSE_CONFIRM based on LLM suggestion. Handle objections if necessary (might need new states).

# STATE: BOOKING_CLOSE_CONFIRM
# System Note: Confirming specific time slot.
Dialogue: Okay, great. So, just to confirm, [Time Suggested by User or AI]? Does that work? [If needed: I have [Time Slot 1] or [Time Slot 2] available then. Which suits you best?]
# App Logic: Listen for confirmation. Store agreed appointment time via CallStateManager. Transition to CONSOLIDATE_DECISIONMAKERS based on LLM suggestion.

# STATE: CONSOLIDATE_DECISIONMAKERS
# System Note: Check for others involved.
Dialogue: Excellent. Just quickly, is there anybody else involved in deciding if these products are right for your business? Can they potentially join the quick 10-minute call?
# App Logic: Listen for response. Store info if applicable. Transition to CONSOLIDATE_DETAILS based on LLM suggestion.

# STATE: CONSOLIDATE_DETAILS
# System Note: Confirm contact info.
Dialogue: Perfect. So just to confirm, your full name is [Owner Name]? And I have the business name and address as [Business Name, Address - if known, otherwise ask]. Is that correct?
# App Logic: Verify/update details in CallStateManager/DB. Transition to CONSOLIDATE_NEXTSTAGE.

# STATE: CONSOLIDATE_NEXTSTAGE
# System Note: Explain appointment logistics.
Dialogue: Great. You will get a text or a quick call to remind you on the day of the appointment, and we will email you a link for the Google Meets video call shortly. Could I just confirm the best mobile number and email address for that? (pause & wait)
# App Logic: Listen for/confirm mobile & email. Store details. Transition to FAREWELL.

# STATE: FAREWELL
# System Note: End the call politely.
Dialogue: Excellent. Well, it's been wonderful speaking with you, [Owner Name]. We look forward to meeting you on the call. Take care now, goodbye.
# App Logic: Trigger call end via Twilio. Log outcome as 'Appointment Booked'.

# STATE: ERROR_HANDLE
# System Note: Fallback state if LLM fails or conversation goes off track.
Dialogue: Apologies, I seem to have gotten a bit lost there. Could you perhaps repeat your last point? # Or: Could we perhaps schedule a quick call back when the owner is available?
# App Logic: Log error, potentially attempt recovery or schedule human callback.