import logging
from typing import Dict, List, Optional
import langdetect
from langdetect import detect
import os

# Make googletrans optional (it pulls httpx which breaks on Python 3.13 due to cgi removal)
try:
    from googletrans import Translator as GoogleTranslator  # type: ignore
except Exception:  # pragma: no cover - optional dependency may fail
    GoogleTranslator = None

logger = logging.getLogger(__name__)

class TranslationService:
    """Service for handling multilingual translation"""
    
    def __init__(self):
        # Initialize translator if available; otherwise operate in pass-through mode
        self.translator = GoogleTranslator() if GoogleTranslator else None
        
        # Supported languages mapping
        self.supported_languages = {
            "en": "English",
            "hi": "Hindi",
            "bn": "Bengali",
            "te": "Telugu",
            "mr": "Marathi",
            "ta": "Tamil",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "pa": "Punjabi",
            "or": "Odia",
            "as": "Assamese"
        }
        
        # Common health terms in different languages
        self.health_terms = {
            "en": {
                "fever": "fever",
                "cough": "cough",
                "headache": "headache",
                "pain": "pain",
                "doctor": "doctor",
                "hospital": "hospital",
                "medicine": "medicine",
                "vaccine": "vaccine"
            },
            "hi": {
                "fever": "बुखार",
                "cough": "खांसी",
                "headache": "सिरदर्द",
                "pain": "दर्द",
                "doctor": "डॉक्टर",
                "hospital": "अस्पताल",
                "medicine": "दवा",
                "vaccine": "टीका"
            },
            "bn": {
                "fever": "জ্বর",
                "cough": "কাশি",
                "headache": "মাথাব্যথা",
                "pain": "ব্যথা",
                "doctor": "ডাক্তার",
                "hospital": "হাসপাতাল",
                "medicine": "ঔষধ",
                "vaccine": "টিকা"
            },
            "te": {
                "fever": "జ్వరం",
                "cough": "కఫం",
                "headache": "తలనొప్పి",
                "pain": "నొప్పి",
                "doctor": "డాక్టర్",
                "hospital": "ఆసుపత్రి",
                "medicine": "మందు",
                "vaccine": "వ్యాక్సిన్"
            }
        }
    
    async def translate_text(self, text: str, target_language: str, source_language: str = None) -> str:
        """Translate text to target language"""
        try:
            if not text or not text.strip():
                return text
            
            # Detect source language if not provided
            if not source_language:
                try:
                    source_language = detect(text)
                except:
                    source_language = "en"  # Default to English
            
            # If source and target are the same, return original text
            if source_language == target_language:
                return text
            
            # If no translator available, return original text (best-effort fallback)
            if not self.translator:
                if source_language != target_language:
                    logger.warning("Translation requested but googletrans is unavailable; returning original text")
                return text

            # Translate the text using googletrans
            result = self.translator.translate(text, src=source_language, dest=target_language)
            return result.text
            
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return text  # Return original text if translation fails
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        try:
            if not text or not text.strip():
                return "en"  # Default to English
            
            detected_lang = detect(text)
            return detected_lang
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return "en"  # Default to English
    
    async def translate_health_terms(self, text: str, target_language: str) -> str:
        """Translate health-specific terms with better accuracy"""
        try:
            if target_language not in self.health_terms:
                return await self.translate_text(text, target_language)
            
            translated_text = text
            source_terms = self.health_terms["en"]  # English terms as source
            target_terms = self.health_terms[target_language]
            
            # Replace health terms with their translations
            for english_term, target_term in target_terms.items():
                if english_term in source_terms:
                    # Case-insensitive replacement
                    import re
                    pattern = re.compile(re.escape(english_term), re.IGNORECASE)
                    translated_text = pattern.sub(target_term, translated_text)
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Error translating health terms: {str(e)}")
            return await self.translate_text(text, target_language)
    
    async def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.supported_languages
    
    async def is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code in self.supported_languages
    
    async def translate_response_template(self, template: str, language: str, **kwargs) -> str:
        """Translate response templates with placeholders"""
        try:
            # Translate the template
            translated_template = await self.translate_text(template, language)
            
            # Replace placeholders if any
            if kwargs:
                translated_template = translated_template.format(**kwargs)
            
            return translated_template
            
        except Exception as e:
            logger.error(f"Error translating response template: {str(e)}")
            return template
    
    async def get_localized_greeting(self, language: str) -> str:
        """Get greeting message in specified language"""
        greetings = {
            "en": "Hello! I'm your health assistant. How can I help you today?",
            "hi": "नमस्ते! मैं आपका स्वास्थ्य सहायक हूं। आज मैं आपकी कैसे मदद कर सकता हूं?",
            "bn": "হ্যালো! আমি আপনার স্বাস্থ্য সহায়ক। আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
            "te": "హలో! నేను మీ ఆరోగ్య సహాయకుడిని। ఈరోజు నేను మీకు ఎలా సహాయపడగలను?",
            "mr": "नमस्कार! मी तुमचा आरोग्य सहायक आहे. आज मी तुम्हाला कशी मदत करू शकतो?",
            "ta": "வணக்கம்! நான் உங்கள் ஆரோக்கிய உதவியாளர். இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?",
            "gu": "નમસ્તે! હું તમારો આરોગ્ય સહાયક છું. આજે હું તમારી કેવી રીતે મદદ કરી શકું?",
            "kn": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಆರೋಗ್ಯ ಸಹಾಯಕ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
            "ml": "ഹലോ! ഞാൻ നിങ്ങളുടെ ആരോഗ്യ സഹായിയാണ്. ഇന്ന് ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കാം?",
            "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ ਸਿਹਤ ਸਹਾਇਕ ਹਾਂ। ਅੱਜ ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
            "or": "ନମସ୍କାର! ମୁଁ ତୁମର ସ୍ୱାସ୍ଥ୍ୟ ସହାୟକ। ଆଜି ମୁଁ ତୁମକୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
            "as": "নমস্কাৰ! মই আপোনাৰ স্বাস্থ্য সহায়ক। আজি মই আপোনাক কেনেকৈ সহায় কৰিব পাৰোঁ?"
        }
        
        return greetings.get(language, greetings["en"])
    
    async def get_localized_emergency_message(self, language: str) -> str:
        """Get emergency message in specified language"""
        emergency_messages = {
            "en": "🚨 EMERGENCY ALERT 🚨\n\nFor immediate medical assistance:\n• Call 108 (Emergency Ambulance)\n• Contact nearest hospital\n• Visit emergency room immediately\n\nStay calm and seek professional help right away.",
            "hi": "🚨 आपातकालीन चेतावनी 🚨\n\nतत्काल चिकित्सा सहायता के लिए:\n• 108 पर कॉल करें (आपातकालीन एम्बुलेंस)\n• निकटतम अस्पताल से संपर्क करें\n• तुरंत आपातकालीन कक्ष में जाएं\n\nशांत रहें और तुरंत पेशेवर सहायता लें।",
            "bn": "🚨 জরুরি সতর্কতা 🚨\n\nতাত্ক্ষণিক চিকিৎসা সহায়তার জন্য:\n• ১০৮ নম্বরে কল করুন (জরুরি অ্যাম্বুলেন্স)\n• নিকটবর্তী হাসপাতালে যোগাযোগ করুন\n• অবিলম্বে জরুরি বিভাগে যান\n\nশান্ত থাকুন এবং অবিলম্বে পেশাদার সাহায্য নিন।",
            "te": "🚨 అత్యవసర హెచ్చరిక 🚨\n\nతక్షణ వైద్య సహాయం కోసం:\n• 108 కి కాల్ చేయండి (అత్యవసర ఆంబులెన్స్)\n• దగ్గరి ఆసుపత్రిని సంప్రదించండి\n• వెంటనే అత్యవసర విభాగానికి వెళ్లండి\n\nశాంతంగా ఉండి వెంటనే వృత్తిపరమైన సహాయం పొందండి।"
        }
        
        return emergency_messages.get(language, emergency_messages["en"])
    
    async def get_localized_vaccination_message(self, language: str, vaccine_name: str) -> str:
        """Get vaccination message in specified language"""
        templates = {
            "en": f"Vaccination Schedule for {vaccine_name}:\n\nPlease consult with your healthcare provider for the most up-to-date vaccination schedule.",
            "hi": f"{vaccine_name} के लिए टीकाकरण कार्यक्रम:\n\nसबसे अद्यतन टीकाकरण कार्यक्रम के लिए कृपया अपने स्वास्थ्य सेवा प्रदाता से परामर्श करें।",
            "bn": f"{vaccine_name} এর জন্য টিকাদান সময়সূচী:\n\nসর্বশেষ টিকাদান সময়সূচীর জন্য অনুগ্রহ করে আপনার স্বাস্থ্যসেবা প্রদানকারীর সাথে পরামর্শ করুন।",
            "te": f"{vaccine_name} కోసం టీకాకరణ షెడ్యూల్:\n\nఅత్యంత తాజా టీకాకరణ షెడ్యూల్ కోసం దయచేసి మీ ఆరోగ్య సేవా ప్రదాతతో సంప్రదించండి।"
        }
        
        return templates.get(language, templates["en"])
