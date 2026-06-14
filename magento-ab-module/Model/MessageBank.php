<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model;

class MessageBank
{
    private const MESSAGES = [
        'browse_indecision' => 'Still comparing options? The item you chose is ready in your cart — secure it now before it sells out.',
        'rapid_browsing' => 'You moved fast for a reason — finish checkout now and your pick is yours.',
        'impulse_add' => 'Almost done — complete your order in a few clicks while everything is reserved.',
        'default' => 'Your cart is ready — complete your order now before items sell out.',
    ];

    public function messageFor(string $signal): string
    {
        return self::MESSAGES[$signal] ?? self::MESSAGES['default'];
    }
}
