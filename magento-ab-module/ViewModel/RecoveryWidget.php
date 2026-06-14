<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\ViewModel;

class RecoveryWidget implements \Magento\Framework\View\Element\Block\ArgumentInterface
{
    private const IDLE_SECONDS = 15;

    private bool $resolved = false;

    /** @var array<string, mixed>|null */
    private ?array $state = null;

    public function __construct(
        private readonly \Bregu\CartRecoveryAb\Model\Config $config,
        private readonly \Bregu\CartRecoveryAb\Model\SessionSignals $signals,
        private readonly \Bregu\CartRecoveryAb\Model\RiskScorer $riskScorer,
        private readonly \Bregu\CartRecoveryAb\Model\AbAssigner $abAssigner,
        private readonly \Bregu\CartRecoveryAb\Model\InterventionProvider $interventionProvider,
        private readonly \Magento\Checkout\Model\Session $checkoutSession,
        private readonly \Magento\Framework\UrlInterface $url,
        private readonly \Magento\Framework\Data\Form\FormKey $formKey,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function isActive(): bool
    {
        return $this->resolve() !== null;
    }

    public function shouldShowBanner(): bool
    {
        $state = $this->resolve();

        return $state !== null && $state['message'] !== null;
    }

    public function getMessage(): string
    {
        $state = $this->resolve();

        return (string) ($state['message'] ?? '');
    }

    public function getClientConfigJson(): string
    {
        $state = $this->resolve();
        if ($state === null) {
            return $this->json->serialize([]);
        }

        return $this->json->serialize([
            'sessionKey' => $state['sessionKey'],
            'bucket' => $state['bucket'],
            'risk' => round((float) $state['risk'], 4),
            'signal' => $state['signal'],
            'trigger' => $this->config->getTrigger(),
            'idleSeconds' => self::IDLE_SECONDS,
            'hasBanner' => $state['message'] !== null,
            'trackUrl' => $this->url->getUrl('cartrecoveryab/ab/track'),
            'formKey' => $this->formKey->getFormKey(),
        ]);
    }

    /**
     * @return array<string, mixed>|null
     */
    private function resolve(): ?array
    {
        if ($this->resolved) {
            return $this->state;
        }
        $this->resolved = true;

        if (!$this->config->isEnabled() || !$this->signals->isFrozen()) {
            return $this->state;
        }

        $quoteId = (int) $this->checkoutSession->getQuoteId();
        if ($quoteId <= 0) {
            return $this->state;
        }

        $views = $this->signals->getViews();
        $events = $this->signals->getEvents();
        $intensity = $this->signals->getBrowseIntensity();
        $risk = $this->riskScorer->abandonmentProbability($views, $events, $intensity);
        $bucket = $this->abAssigner->bucketFor((string) $quoteId);
        $signal = $this->dominantSignal($views, $intensity);

        $message = null;
        if ($bucket === \Bregu\CartRecoveryAb\Model\AbAssigner::BUCKET_TREATMENT
            && $risk >= $this->config->getRiskThreshold()
        ) {
            $message = $this->interventionProvider->getMessage($signal, $this->context($views, $intensity));
        }

        $this->state = [
            'sessionKey' => (string) $quoteId,
            'bucket' => $bucket,
            'risk' => $risk,
            'signal' => $signal,
            'message' => $message,
        ];

        return $this->state;
    }

    /**
     * @return array{cart_value: float, cart_items: int, views: int, browse_intensity: float}
     */
    private function context(int $views, float $intensity): array
    {
        $quote = $this->checkoutSession->getQuote();

        return [
            'cart_value' => (float) $quote->getGrandTotal(),
            'cart_items' => (int) $quote->getItemsQty(),
            'views' => $views,
            'browse_intensity' => $intensity,
        ];
    }

    private function dominantSignal(int $views, float $intensity): string
    {
        if ($views >= 6) {
            return 'browse_indecision';
        }
        if ($intensity >= 5.0) {
            return 'rapid_browsing';
        }
        if ($views <= 1) {
            return 'impulse_add';
        }

        return 'default';
    }
}
