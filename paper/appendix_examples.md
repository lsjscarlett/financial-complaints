# Appendix D. Worked examples

Two complaints from the corpus with all six replies. Narratives are shown as published by the CFPB (redactions as `XXXX` / `{$...}`).


## D.1 Complaint 4083: Debt collection / Electronic communications / Contacted before 8am or after 9pm

> Debt collection agency texted me at XXXX XXXX. Text message woke me up in the middle of the night. I called the agency during business hours and a representative confirmed the text was sent at XXXX XXXX but was otherwise dismissive. Highly inappropriate time to text someone. Disrupted my sleep. I wake up early for work and suffer from bouts of XXXX. After receiving this text message I was unable to return to sleep, resulting in an exhausted, unproductive work day. This has impacted my income for the day, as I work on commission and was too tired to perform my work duties efficiently.


**V1 terse, ChatGPT** (VADER +0.38)

> We sincerely apologize for the inconvenience and discomfort caused by our late-night communication. Your experience is important to us, and we will address this issue internally to ensure it does not happen again in the future.


**V1 terse, Mistral** (VADER +0.56)

> We sincerely apologize for the inconvenience and disruption caused by the untimely text message. We take your concerns seriously and will address this matter internally to ensure our communications comply with all applicable regulations and respect your personal time.


**V2 empathetic, ChatGPT** (VADER +0.66, contains placeholder)

> Dear [Customer's Name],
> 
> I’m really sorry to hear about the late-night text you received and the disruption it caused to your sleep and work. I understand how frustrating this must be, and I want to assure you that we take this matter seriously. I will personally reach out to the debt collection agency to address their messaging practices and ensure they adhere to appropriate contact times in the future. Thank you for bringing this to our attention, and please know that we’re here to support you.


**V2 empathetic, Mistral** (VADER +0.48, contains placeholder)

> I’m truly sorry this happened—being woken up like that is frustrating, and I completely understand how much it disrupted your day. This isn’t the level of service we expect from our partners, and I’ll be reaching out to the debt collection agency immediately to address this and ensure it doesn’t happen again. You’ll hear from me again by [specific date, e.g., Friday] with an update on what we’ve done to fix this.


**V3 structured, ChatGPT** (VADER -0.45)

> Acknowledgement: We understand that you received a text message from a debt collection agency at an inappropriate hour, which disrupted your sleep and affected your workday.  
> Next step: We will investigate this matter and review the text messaging practices of the agency involved; you can expect an update from us within the next 10 business days.  
> What we need from you: Nothing at this time.


**V3 structured, Mistral** (VADER +0.03)

> **Acknowledgement:** We understand that you received an electronic communication from our debt collection agency outside of the permitted hours.
> 
> **Next step:** We will review our procedures to ensure compliance with communication timing requirements and follow up with you within 10 business days.
> 
> **What we need from you:** Nothing at this time.


## D.2 Complaint 7368: Mortgage / Struggling to pay mortgage / Trying to communicate with the company to fix an issue related to modification, forbearance, short sale, deed-in-lieu, bankruptcy, or foreclosure

> To start I got behind because my loan was sold to Carrington, after I caught it up, Do to my financial situation the last month and this up coming month I am having to pay just a little over 30 days past the due date, however even though I am caught up to this month, they threaten me with foreclose and it wa only 45 days, with the payment already scheduled! I had nothing but stress, threats and issues with carrington and harassing phone calls, now a letter saying notice of intent to foreclose!!!! I am so stress that I dont what else to do


**V1 terse, ChatGPT** (VADER +0.53)

> We understand how stressful this situation is and sincerely apologize for any distress caused by our communication. Please reach out to our customer service team directly so we can work together to find a resolution that best fits your financial needs and clarify any misunderstandings regarding your account.


**V1 terse, Mistral** (VADER +0.08)

> I’m truly sorry to hear about the stress and frustration you’re experiencing with Carrington. Please know that we’re here to help—let’s work together to find a solution, whether it’s adjusting your payment plan, exploring forbearance options, or reviewing your loan modification request.


**V2 empathetic, ChatGPT** (VADER +0.84)

> I’m really sorry to hear about the stress and frustration you’re experiencing with your mortgage situation. I understand how overwhelming it can be to receive those kinds of communications when you’re already trying to manage your payments. I will personally reach out to our team to ensure your case is reviewed and we address the notice of intent to foreclose. We want to help you find the best solution, so please know you’re not alone in this.


**V2 empathetic, Mistral** (VADER -0.07)

> I’m so sorry you’ve been through this—it sounds incredibly stressful, and I want to assure you we’ll do everything we can to help. I’ll personally escalate this to our loss mitigation team today to review your account and pause any foreclosure actions while we work on a solution. You should hear back from us within 48 hours with next steps.


**V3 structured, ChatGPT** (VADER -0.03)

> Acknowledgement: We understand that you are experiencing significant stress regarding your mortgage payments and have received threats of foreclosure despite being caught up on your payments.  
> Next step: We will review your account and the details of your situation to provide the best possible options for addressing your concerns, and you should hear from us within the next five business days.  
> What we need from you: Nothing at this time.


**V3 structured, Mistral** (VADER -0.72)

> **Acknowledgement:** We understand you are experiencing stress due to concerns about foreclosure threats and communication issues with Carrington.
> 
> **Next step:** We will review your account and reach out to you within 10 business days to discuss available options.
> 
> **What we need from you:** Nothing at this time.
