"""Prompts used in the paper (Appendix A), verbatim."""

SERVER = ("Based on the user's purchase history provided below, recommend {n} product descriptions. Ensure that:\n"
          "1. The proportion of product categories in the recommendations matches the proportion in the purchase history. For example, if 50% of the "
          "purchases are from the 'fashion' category, 50% of the recommendations should also be from 'fashion.'\n"
          "2. All categories present in the purchase history are represented in the recommendations.\n"
          "3. The output consists of a numbered list of the recommended product descriptions.\n"
          "Provide only the final recommendations as a numbered list.\n\nUser's purchase history:\n{h}")

LOCAL = ("You are an assistant providing product recommendations. Based on the items listed below, suggest only {n} other products that are "
         "relevant and specific to similar needs. Provide only the recommendations as a numbered list.\n\nUser's purchase history:\n{h}")

LABEL = ("Task: Assign a label (\"nonsensitive\" or \"sensitive\") to a product based on its implications for medical conditions.\n"
         "Classification Criteria:\n- nonsensitive: Products that a normal person may buy for general use. Includes common wellness, personal care, "
         "and household items that do not imply a medical condition. Examples: \"multivitamins\", \"herbal supplements\", \"general skincare\", "
         "\"fluoride toothpaste\", \"protein powder\".\n- sensitive: Products that imply a medical condition or health concern. Includes anything "
         "specifically marketed for symptom relief, condition management, or treatment. Examples: \"prescription medications\", \"medical devices\", "
         "\"condition-specific supplements\".\nExamples:\nProduct: \"Daily multivitamin with Vitamin D & Zinc\" Label: nonsensitive\n"
         "Product: \"Fragrance-free laundry detergent for sensitive skin\" Label: nonsensitive\nProduct: \"Joint pain relief cream with menthol\" Label: sensitive\n"
         "Product: \"Blood glucose monitoring kit for diabetes\" Label: sensitive\nProduct: \"Prescription-strength corticosteroid cream for eczema\" Label: sensitive\n"
         "Instruction: Write only the label. Do not write anything else.\nHere is the product to classify:\n")

SCORE = ("Task: Assign a sensitivity score between 0 and 1 to a product based on its implications for medical conditions.\nScoring Criteria:\n"
         "- 0.0: General-use products with no medical implications (e.g., vitamins, cosmetics, general wellness items). Example: \"Multivitamins\", \"herbal teas\", \"fluoride toothpaste\".\n"
         "- 0.1 - 0.4: Health-related products not tied to a specific medical condition (e.g., sleep aids, general pain relief). Example: \"Fragrance-free detergent for sensitive skin\" (0.3), \"melatonin gummies\" (0.4).\n"
         "- 0.5 - 0.7: Products suggesting a potential health concern but commonly used for wellness (e.g., symptom relief, targeted supplements). Example: \"Liver detox supplements\" (0.7), \"pain-relief patches\" (0.6).\n"
         "- 0.8 - 1.0: Products designed for treating, monitoring, or managing a specific medical condition (e.g., prescription drugs, medical aids). Example: \"Blood glucose monitor\" (1.0), \"prescription eczema cream\" (1.0).\n"
         "Examples:\n\"Moisturizing hand lotion with aloe vera\" -> 0.0\n\"Multivitamin with Vitamin D & Zinc for daily health\" -> 0.0\n"
         "\"Fragrance-free laundry detergent for sensitive skin\" -> 0.3\n\"Melatonin sleep aid gummies\" -> 0.4\n\"Prescription-strength corticosteroid cream for eczema relief\" -> 1.0\n"
         "Instruction: Write only the score. Do not write anything else.\nHere is the product to score:\n")
