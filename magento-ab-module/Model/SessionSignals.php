<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model;

class SessionSignals
{
    private const KEY_VIEWS = 'bregu_views_before_cart';
    private const KEY_EVENTS = 'bregu_events_before_cart';
    private const KEY_START = 'bregu_session_start';
    private const KEY_INTENSITY = 'bregu_browse_intensity';
    private const KEY_FROZEN = 'bregu_signals_frozen';

    public function __construct(
        private readonly \Magento\Checkout\Model\Session $checkoutSession
    ) {
    }

    public function recordView(int $timestamp): void
    {
        if ($this->isFrozen()) {
            return;
        }

        $session = $this->checkoutSession;
        if ($session->getData(self::KEY_START) === null) {
            $session->setData(self::KEY_START, $timestamp);
        }
        $session->setData(self::KEY_VIEWS, (int) $session->getData(self::KEY_VIEWS) + 1);
        $session->setData(self::KEY_EVENTS, (int) $session->getData(self::KEY_EVENTS) + 1);
    }

    public function freezeAtCartAdd(int $timestamp): void
    {
        if ($this->isFrozen()) {
            return;
        }

        $session = $this->checkoutSession;
        $start = (int) ($session->getData(self::KEY_START) ?? $timestamp);
        $elapsedMinutes = max(1, $timestamp - $start) / 60.0;

        $session->setData(self::KEY_INTENSITY, (int) $session->getData(self::KEY_VIEWS) / $elapsedMinutes);
        $session->setData(self::KEY_FROZEN, true);
    }

    public function getViews(): int
    {
        return (int) $this->checkoutSession->getData(self::KEY_VIEWS);
    }

    public function getEvents(): int
    {
        return (int) $this->checkoutSession->getData(self::KEY_EVENTS);
    }

    public function getBrowseIntensity(): float
    {
        return (float) $this->checkoutSession->getData(self::KEY_INTENSITY);
    }

    public function isFrozen(): bool
    {
        return (bool) $this->checkoutSession->getData(self::KEY_FROZEN);
    }
}
