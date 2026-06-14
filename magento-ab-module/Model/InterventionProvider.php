<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model;

class InterventionProvider
{
    public function __construct(
        private readonly \Bregu\CartRecoveryAb\Model\Config $config,
        private readonly \Bregu\CartRecoveryAb\Service\GeminiClient $geminiClient,
        private readonly \Bregu\CartRecoveryAb\Model\MessageBank $messageBank
    ) {
    }

    /**
     * @param array{cart_value: float, cart_items: int, views: int, browse_intensity: float} $context
     */
    public function getMessage(string $signal, array $context): string
    {
        if ($this->config->getInterventionSource() === \Bregu\CartRecoveryAb\Model\Config::SOURCE_LIVE) {
            $generated = $this->geminiClient->generate($this->buildPrompt($signal, $context));
            if ($generated !== null && $generated !== '') {
                return $generated;
            }
        }

        return $this->messageBank->messageFor($signal);
    }

    /**
     * @param array{cart_value: float, cart_items: int, views: int, browse_intensity: float} $context
     */
    private function buildPrompt(string $signal, array $context): string
    {
        return "You are an e-commerce cart-recovery copywriter.\n\n"
            . "USER SESSION (right at the moment they added the item to cart):\n"
            . '- Cart value: $' . number_format($context['cart_value'], 2) . "\n"
            . '- Cart items: ' . $context['cart_items'] . "\n"
            . '- Views before cart: ' . $context['views'] . "\n"
            . '- Browse intensity: ' . number_format($context['browse_intensity'], 2) . " clicks/min\n"
            . '- Dominant abandonment signal: ' . $signal . "\n\n"
            . "TASK: write ONE persuasive single-sentence pop-up message (max 25 words) that\n"
            . "- addresses this user's specific risk signal,\n"
            . "- is concrete to their cart,\n"
            . "- creates a gentle urgency,\n"
            . "- reads as natural human copy — no hashtags, no emoji, no ALL CAPS, no quotes.\n\n"
            . 'Output only the single sentence, nothing else.';
    }
}
